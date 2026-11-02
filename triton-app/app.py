import io
import json
import logging
import os
import warnings
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 16_000_000
LABELS = [
    line.split(" ", 1)[1]
    for line in Path(__file__).with_name("synset.txt").read_text().splitlines()
]
if len(LABELS) != 1000:
    raise RuntimeError("Expected 1000 ImageNet class labels")
TRITON_URL = os.environ.get("TRITON_URL", "http://localhost:8000").rstrip("/")


def preprocess(data):
    if len(data) > 5 * 1024 * 1024:
        raise ValueError("Each image must be at most 5 MiB")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in ("JPEG", "PNG", "WEBP"):
                    raise ValueError("Use a JPEG, PNG or WebP image")
                if source.width * source.height > Image.MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds 16 million pixels")
                image = ImageOps.exif_transpose(source).convert("RGB")
                w, h = image.size
                scale = 256 / min(w, h)
                # Bound allocation for unusually long, narrow images.
                resized = (round(w * scale), round(h * scale))
                if resized[0] * resized[1] > Image.MAX_IMAGE_PIXELS:
                    raise ValueError("Image aspect ratio is too extreme")
                image = image.resize(resized, Image.Resampling.BILINEAR)
                left, top = (image.width - 224) // 2, (image.height - 224) // 2
                image = image.crop((left, top, left + 224, top + 224))
                array = np.asarray(image, dtype=np.float32) / 255
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            raise ValueError("Invalid, damaged or oversized image") from exc
    array = (array - np.array([.485, .456, .406], dtype=np.float32)) / np.array(
        [.229, .224, .225], dtype=np.float32
    )
    return array.transpose(2, 0, 1)


def classify(tensors):
    batch = np.stack(tensors).astype("<f4")
    binary = batch.tobytes(order="C")
    payload = {
        "inputs": [{"name": "data", "shape": list(batch.shape),
                    "datatype": "FP32", "parameters": {"binary_data_size": len(binary)}}],
        "outputs": [{"name": "resnetv24_dense0_fwd"}],
    }
    header = json.dumps(payload).encode()
    req = Request(
        TRITON_URL + "/v2/models/resnet50/infer",
        data=header + binary,
        headers={"Content-Type": "application/octet-stream",
                 "Inference-Header-Content-Length": str(len(header))},
    )
    with urlopen(req, timeout=30) as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise RuntimeError("Triton response exceeds expected size")
    result = json.loads(raw)
    output = next(
        (o for o in result.get("outputs", [])
         if o.get("name") == "resnetv24_dense0_fwd"), None
    )
    if output is None or output.get("shape") != [len(tensors), 1000] or output.get("datatype") != "FP32":
        raise RuntimeError("Unexpected Triton output contract")
    scores = np.asarray(output["data"], dtype=np.float64)
    if scores.size != len(tensors) * 1000 or not np.isfinite(scores).all():
        raise RuntimeError("Invalid Triton scores")
    scores = scores.reshape(len(tensors), 1000)
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return [
        [{"class_id": int(i), "label": LABELS[i], "score": float(row[i])}
         for i in np.argsort(row)[-5:][::-1]]
        for row in probabilities
    ]


@app.get("/")
def index():
    return render_template("index.html", labels=LABELS)


@app.get("/healthz")
def health():
    return jsonify(status="ok")


@app.get("/readyz")
def ready():
    try:
        with urlopen(TRITON_URL + "/v2/models/resnet50/ready", timeout=3):
            pass
    except (OSError, HTTPError, URLError):
        app.logger.warning("Triton model readiness check failed")
        return jsonify(error="ResNet-50 is not reachable or ready"), 503
    return jsonify(status="ready")


@app.post("/api/classify")
def infer():
    try:
        uploads = request.files.getlist("images")
        if not 1 <= len(uploads) <= 2:
            raise ValueError("Upload one or two images")
        expected = request.form.get("expected", "")
        if expected and (not expected.isdigit() or not 0 <= int(expected) < 1000):
            raise ValueError("Select a valid expected class")
        tensors = [preprocess(f.read(5 * 1024 * 1024 + 1)) for f in uploads]
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    try:
        predictions = classify(tensors)
    except (OSError, HTTPError, URLError, ValueError, RuntimeError, KeyError, TypeError):
        app.logger.exception("Triton inference failed")
        return jsonify(error="Triton inference failed. Check model readiness and server logs."), 502
    results = []
    for top in predictions:
        results.append({
            "top5": top,
            "expected_top1_match": top[0]["class_id"] == int(expected) if expected else None,
            "expected_top5_match": any(x["class_id"] == int(expected) for x in top) if expected else None,
        })
    return jsonify(
        results=results,
        same_predicted_category=(predictions[0][0]["class_id"] == predictions[1][0]["class_id"]) if len(predictions) == 2 else None,
        note="Category comparison only. This does not verify the same object or person. Scores are not calibrated confidence.",
    )


@app.errorhandler(RequestEntityTooLarge)
def oversized(_error):
    return jsonify(error="Total upload exceeds 12 MiB"), 413


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=8080)
