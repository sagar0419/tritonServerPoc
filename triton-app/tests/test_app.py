import io
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
from PIL import Image

from app import app, classify, preprocess


def image_bytes(color="red"):
    data = io.BytesIO()
    Image.new("RGB", (300, 260), color).save(data, "PNG")
    return data.getvalue()


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_preprocess_contract_and_normalization(self):
        array = preprocess(image_bytes())
        self.assertEqual(array.shape, (3, 224, 224))
        self.assertEqual(array.dtype, np.float32)
        np.testing.assert_allclose(
            array[:, 0, 0], (np.array([1, 0, 0]) - [.485, .456, .406]) / [.229, .224, .225],
            rtol=1e-6,
        )

    def test_invalid_uploads_rejected(self):
        for payload in (
            {},
            {"images": (io.BytesIO(b"not an image"), "bad.png")},
            {"images": (io.BytesIO(image_bytes()), "image.png"), "expected": "1000"},
            {"images": [(io.BytesIO(image_bytes()), "a.png")] * 3},
        ):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/api/classify", data=payload).status_code, 400)

    @patch("app.classify")
    def test_comparison_and_expected_matches(self, infer):
        infer.return_value = [
            [{"class_id": 1, "label": "goldfish", "score": .8},
             {"class_id": 2, "label": "shark", "score": .2}],
            [{"class_id": 1, "label": "goldfish", "score": .9}],
        ]
        result = self.client.post("/api/classify", data={
            "images": [(io.BytesIO(image_bytes()), "a.png"), (io.BytesIO(image_bytes()), "b.png")],
            "expected": "2",
        })
        self.assertEqual(result.status_code, 200)
        data = result.get_json()
        self.assertTrue(data["same_predicted_category"])
        self.assertFalse(data["results"][0]["expected_top1_match"])
        self.assertTrue(data["results"][0]["expected_top5_match"])
        infer.return_value[1][0]["class_id"] = 3
        result = self.client.post("/api/classify", data={
            "images": [(io.BytesIO(image_bytes()), "a.png"), (io.BytesIO(image_bytes()), "b.png")]
        })
        self.assertFalse(result.get_json()["same_predicted_category"])

    @patch("app.classify", side_effect=URLError("unavailable"))
    def test_upstream_failure_not_reported_as_match(self, _infer):
        result = self.client.post("/api/classify", data={
            "images": (io.BytesIO(image_bytes()), "image.png")
        })
        self.assertEqual(result.status_code, 502)
        self.assertIn("error", result.get_json())

    @patch("app.urlopen")
    def test_output_validation(self, upstream):
        upstream.return_value.__enter__.return_value.read.return_value = b'{"outputs":[]}'
        with self.assertRaises(RuntimeError):
            classify([preprocess(image_bytes())])

    @patch("app.urlopen")
    def test_binary_transport_and_score_ordering(self, upstream):
        scores = [0.] * 1000
        scores[258] = 10.
        upstream.return_value.__enter__.return_value.read.return_value = json.dumps({
            "outputs": [{"name": "resnetv24_dense0_fwd", "datatype": "FP32",
                         "shape": [1, 1000], "data": scores}]
        }).encode()
        tensor = preprocess(image_bytes())
        result = classify([tensor])
        self.assertEqual(result[0][0]["class_id"], 258)
        req = upstream.call_args.args[0]
        length = int(req.get_header("Inference-header-content-length"))
        metadata = json.loads(req.data[:length])
        self.assertEqual(metadata["inputs"][0]["parameters"]["binary_data_size"], tensor.nbytes)
        self.assertEqual(req.data[length:], tensor.astype("<f4").tobytes())


if __name__ == "__main__":
    unittest.main()
