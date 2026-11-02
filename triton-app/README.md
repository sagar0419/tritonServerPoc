# Triton Vision Lab app

CPU-only browser UI for the existing ResNet-50 Triton service.
Upload one or two images, view the top-five predictions, select an expected
ImageNet class, and see top-1/top-5 matches. Two images are compared by their
predicted top-1 class. **This is not same-object or person identity verification.**
Scores are softmax class scores, not calibrated probabilities of correctness.
The model only recognizes ImageNet's 1000 categories; there is no reliable
"unknown object" detector.

## Run locally with your existing Triton port-forward

The expected-class dropdown now has a case-insensitive search box. Type part
of a class name, then select from the filtered, scrollable dropdown. Clear the
search to restore all classes. A selection excluded by a new search is cleared
with a visible message. Published release **1.0.2** includes this search UI;
`kubernetes.yaml` is pinned to its verified digest:
`sha256:c44b37d4b9203b669c9650e88ae72180aaab42696cf1194e0cc94960eec65824`.
Six container tests, the image's rendered HTML and the search behavior test
passed before publication. The manifest was updated locally, not applied.
To try your local source without publishing a new image:

```bash
cd /Users/pasa4023/llm/triton-app
docker build --platform linux/amd64 -t triton-vision-app:search .
docker run --rm --platform linux/amd64 \
  -p 127.0.0.1:8080:8080 \
  -e TRITON_URL=http://host.docker.internal:8000 \
  triton-vision-app:search
```

Open http://localhost:8080. Stop any older app bound to port 8080 first.
Search behavior can be tested with
`node tests/test_class_search.js templates/index.html`.

Triton must expose `http://localhost:8000/v2/models/resnet50/ready`.
The app requires model `resnet50`, input `data` FP32 `[B,3,224,224]`, output
`resnetv24_dense0_fwd` FP32 `[B,1000]`. This was checked against live metadata.
Do not point port 8000 at the app itself.

```bash
docker run --rm --platform linux/amd64 \
  --name triton-vision-app \
  --read-only --tmpfs /tmp:rw,nosuid,size=134217728 \
  -p 127.0.0.1:8080:8080 \
  -e TRITON_URL=http://host.docker.internal:8000 \
  sagar27/appforvisonmodel:1.0.2
```

Open **http://localhost:8080** in your browser. Docker Desktop's
`host.docker.internal` reaches the host port-forward; Linux hosts may need
`--add-host host.docker.internal:host-gateway` and a reachable host listener.
Stop the foreground container with Ctrl-C. The validation helper was stopped;
no local app container is intentionally left running.

Use JPEG, PNG or WebP. Maximum is 5 MiB per file, 16 million decoded pixels,
two files, and 12 MiB total request. Very extreme aspect ratios are rejected.
Pick an expected class only if it corresponds to an ImageNet category.
For example, two different Samoyed dogs can yield category MATCH without being
the same dog.

Images are decoded in memory, EXIF-oriented, converted to RGB, resized with
short side 256, center-cropped to 224, normalized using ImageNet mean/std,
and transposed to CHW. Tensors go to Triton in little-endian FP32 binary format,
avoiding large JSON input arrays. Two images use a single batch-2 request.
HTTP inference timeout is 30 seconds; failures return explicit 502 errors.
Uploads are not saved or included in the published image.

## Kubernetes deployment (commands for you, not executed)

No cluster resources have been created or changed by this task.
After owner approval and confirming the `day-gpu-test` target and namespace:

```bash
kubectl --context day-gpu-test.in.enterprisenet.org \
  apply -f kubernetes.yaml
kubectl --context day-gpu-test.in.enterprisenet.org -n default \
  rollout status deployment/triton-vision-app --timeout=3m
kubectl --context day-gpu-test.in.enterprisenet.org -n default \
  port-forward service/triton-vision-app 8080:8080
```

Open http://localhost:8080. The app calls
`http://triton-server.default.svc.cluster.local:8000` internally, so the cluster
app does not need your local Triton port-forward. It uses no GPU and adds no
Triton replicas. Readiness requires ResNet-50 readiness; liveness only checks
the app process to avoid restarting it when Triton is unavailable.
The image is linux/amd64, approximately 308 MiB as reported by Docker (not
compressed download size). A node's runtime still needs image-unpack headroom.
Actual Kubernetes scheduling, admission and app-to-Triton networking are untested.
Confirm the image repository is pullable by your nodes; if private, use an
approved imagePullSecret without putting credentials into these files.

**Learning-only exposure:** ClusterIP and loopback port-forward are intentional.
No authentication, public ingress, TLS or calibrated identity matcher is supplied.
Do not expose publicly or upload sensitive images. Before broader use, add
approved authentication/TLS, network policy, rate limiting, concurrency budgets,
image/dependency scanning, and monitoring. Upload limits bound individual requests
but are not protection against sustained abusive traffic.

If a rollout fails, inspect readiness, logs and Triton availability rather than
restarting the inference engine. To roll back a later app release, review its
history and use an approved `kubectl rollout undo deployment/triton-vision-app`
in the explicit target context/namespace. There is no previous revision on a
first install. Stopping the local app does not stop Triton.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Upload API, preprocessing, binary Triton client, result comparison and health checks |
| `templates/index.html` | Image-upload UI, previews, expected-class dropdown and results |
| `synset.txt` | ONNX Model Zoo's ImageNet class order |
| `NOTICE`, `LICENSE-ONNX` | Attribution/license for bundled labels |
| `requirements.txt` | App dependencies; no PyTorch or local GPU engine |
| `Dockerfile`, `.dockerignore` | Non-root, CPU-only Linux image; selective source copying |
| `kubernetes.yaml` | Digest-pinned Deployment and internal Service, not applied |
| `tests/test_app.py` | Six preprocessing, binary transport, comparison and error-path tests |
| `plan/001-image-app.md` | Approved feature scope and verification approach |

## Test and rebuild

From this folder:

```bash
docker buildx build --platform linux/amd64 --load \
  -t sagar27/appforvisonmodel:1.0.1 .
docker run --rm --platform linux/amd64 \
  -v "$PWD/tests:/app/tests:ro" \
  sagar27/appforvisonmodel:1.0.1 \
  python -m unittest discover -s tests -v
```

For future code changes, choose a new version tag, test it locally against
Triton, push it, and update `kubernetes.yaml` with the verified registry digest.
The base image is digest-pinned; direct dependencies are pinned, but their
transitive dependencies are not completely locked.

## Verification record

Tested on 2026-11-01 IST against your forwarded live Triton:

- Six unit tests passed inside the final linux/amd64 Docker image.
- Non-root container ran with read-only root filesystem and `/tmp` tmpfs.
- HTML UI endpoint, health and real-model readiness returned HTTP 200.
- Public PyTorch Hub dog fixture classified as **Samoyed, class 258 (~93.9% score)**.
- Two copies returned category MATCH and expected-class top-1/top-5 MATCH.
- Expected class 0 returned NO MATCH; malformed uploads/too many files/invalid
  expected class returned HTTP 400.
- Binary tensor transport fixed a live batch-2 timeout observed with initial
  JSON input transport. Final observed response times were approximately 2.1 s
  for one image and 4.3 s for two over Docker/port-forward; these are integration
  observations, not model execution latency or a performance benchmark.
- Tag **1.0.1** pushed and registry platform/digest verified:
  `sha256:62b9c68eb54cb382df35ed8fd617107924402ae57957a8382b51bbb2b9312612`.

An earlier `1.0.0` tag was published during testing before the batch-2 timeout
was resolved. **Use 1.0.1**, not that earlier tag. No `latest` tag was changed.
Browser click automation, broad accuracy evaluation, vulnerability scans and
Kubernetes deployment were not performed. Local Docker API/HTML tests demonstrate
the real inference path; they do not establish production readiness.

Sources:
- [ONNX Model Zoo ResNet preprocessing and outputs](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/README.md)
- [ONNX Model Zoo class labels](https://github.com/onnx/models/blob/main/validated/vision/classification/synset.txt)
- [Triton binary tensor protocol](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_binary_data.html)

Ownership: you own deployment approval and learning use; the platform owner
owns cluster admission/storage/access; the app owner must review security and
operational controls before broader exposure.
