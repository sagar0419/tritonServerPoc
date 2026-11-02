# Triton Vision PoC

A proof of concept that serves a **ResNet-50 (ONNX)** image classifier with **NVIDIA Triton Inference Server** on a Kubernetes GPU cluster. It measures what Triton's **dynamic batching** and **model instance count** do to throughput and latency, and includes a small CPU-only web app to try the model from a browser.

This is a learning PoC, not a production deployment. See [Limitations](#limitations).

## Architecture

```
Browser -> Flask app (CPU, triton-app) -> Triton Server (GPU, triton-server) -> ResNet-50 ONNX
                                                  |
                                          Prometheus (PodMonitor) -> Grafana
```

- The **app** decodes and preprocesses images on the CPU, then sends a binary FP32 tensor to Triton.
- **Triton** owns the GPU, batches requests and runs the model.
- **perf_analyzer** runs as an in-cluster Job to benchmark Triton directly.

## Repository layout

```
triton-vision/
├── kustomization.yaml              # deploys triton-server + triton-app together
├── GPU-1-resnet50-perf.csv         # perf_analyzer log, instance_group count: 1
├── GPU-2-resnet50-perf.csv         # perf_analyzer log, instance_group count: 2
├── model_repository/               # local draft model repo (see note below)
│   └── image_detection/
│       ├── config.pbtxt
│       └── 1/model.onnx            # empty placeholder file
├── triton-server/
│   ├── kustomization.yaml
│   ├── triton-deployment.yaml      # ConfigMap + Triton Deployment + Service
│   ├── perf-analyzer.yaml          # perf_analyzer Job
│   ├── triton-pod-monitor.yaml     # Prometheus PodMonitor
│   ├── triton-grafana-dashboard.json
│   └── cm.yaml                     # alternate ConfigMap (not in kustomization)
└── triton-app/
    ├── kustomization.yaml
    ├── kubernetes-app.yaml         # app Deployment + ClusterIP Service
    ├── app.py                      # Flask app: upload, preprocess, call Triton
    ├── templates/index.html        # upload UI
    ├── synset.txt                  # ImageNet class labels
    ├── tests/                      # unit tests + class-search UI test
    ├── Dockerfile, .dockerignore, requirements.txt
    ├── NOTICE, LICENSE-ONNX        # label attribution
    └── README.md                   # app-specific details
```

> The deployed model is `resnet50`. `triton-deployment.yaml` downloads `resnet50-v2-7.onnx` from the ONNX Model Zoo in an init container and mounts the config from a ConfigMap. `model_repository/` and `cm.yaml` describe a differently named `image_detection` draft and are **not** what is deployed.

## The Triton model configuration

Defined in the `triton-resnet50-config` ConfigMap inside `triton-server/triton-deployment.yaml`:

| Setting | Value | Meaning |
| --- | --- | --- |
| `name` / `platform` | `resnet50` / `onnxruntime_onnx` | Model name and ONNX Runtime backend |
| `max_batch_size` | `32` | Largest batch Triton may build |
| input | `data`, FP32, `[3, 224, 224]` | Batch dimension is added by Triton |
| output | `resnetv24_dense0_fwd`, FP32, `[1000]` | One score per ImageNet class |
| `instance_group` | `count: 1`, `KIND_GPU` | Model copies on the GPU (tested with 1 and 2) |
| `dynamic_batching` | preferred `[8, 16, 32]`, delay `5000` us | Wait up to 5 ms to fill a batch |

Triton runs with `--strict-model-config=false` and exposes HTTP `8000`, gRPC `8001` and metrics `8002`. The GPU request uses NVIDIA GPU limits. Adjust them for your cluster.

## Prerequisites

- A Kubernetes cluster with a GPU node and the GPU scheduler (or edit the GPU resource limits and `schedulerName` for your setup).
- `kubectl` and `kustomize` support (`kubectl apply -k`).
- Prometheus Operator, only if you use the PodMonitor and Grafana dashboard.
- Docker, only if you rebuild the app image.

## Deploy

Set your own context and namespace first. The manifests use the `default` namespace.

```bash
kubectl --context <context> apply -k .
kubectl --context <context> -n default rollout status deployment/triton-server --timeout=10m
kubectl --context <context> -n default rollout status deployment/triton-vision-app --timeout=3m
```

`triton-server/kustomization.yaml` also includes `perf-analyzer.yaml`, so applying it starts the benchmark Job. Remove that line if you want to start the benchmark manually, and note that the Job can fail if it starts before Triton is ready.

Check the model is ready:

```bash
kubectl -n default port-forward svc/triton-server 8000:8000
curl -s localhost:8000/v2/models/resnet50/ready
```

## Try the app

```bash
kubectl -n default port-forward svc/triton-vision-app 8080:8080
```

Open <http://localhost:8080>, upload one or two JPEG, PNG or WebP images, and view the top-five predictions. The app compares **predicted ImageNet categories only**. It does not verify the same object or person. See [triton-app/README.md](triton-app/README.md) for limits, endpoints and how to rebuild the image.

## Run the benchmark

`triton-server/perf-analyzer.yaml` runs NVIDIA's `perf_analyzer` against `triton-server:8000`:

- model `resnet50`, HTTP, batch size 1 per request, random input
- concurrency sweep 1 to 32, 6-second measurement windows (see the file for the exact flags)

```bash
kubectl -n default apply -f triton-server/perf-analyzer.yaml
kubectl -n default logs -f job/triton-perf-analyzer
```

Delete the Job before re-running it. The two `GPU-*-resnet50-perf.csv` files are saved logs of this Job (plain text despite the `.csv` extension), one with `count: 1` and one with `count: 2`, on a full GPU.

## Results summary

Single run per setting, random input, batch size 1, HTTP, full GPU:

| Clients | 1 instance (infer/s) | 2 instances (infer/s) |
| --- | --- | --- |
| 1 | 128 | 129 |
| 8 | 1,134 | 1,121 |
| 16 | 1,811 | 1,993 |
| 24 | 1,742 | 2,304 |
| 32 | 2,174 | 2,327 |

- At one client, about 5 ms of a 7.8 ms request is queue time, from the 5 ms batching wait.
- Throughput jumps at 8 clients, which matches the first `preferred_batch_size`.
- A second instance helps mid-range (+32% at 24 clients) but only about 7% at 32 clients.
- The one-client baseline includes the batching wait, so it is not a pure no-batching number. GPU utilization was not measured directly.

## Monitoring

- `triton-pod-monitor.yaml` makes Prometheus scrape Triton's `metrics` port every 15 s. It assumes a Prometheus Operator that accepts PodMonitors from other namespaces.
- Import `triton-grafana-dashboard.json` into Grafana for scrape status, request rate, failures, queue time, average batch size and optional GPU panels.

## Limitations

- No authentication, TLS, rate limiting or network policy. Services are `ClusterIP` and meant for port-forward use only. Do not expose them publicly or upload sensitive images.
- ResNet-50 recognizes only the 1,000 ImageNet classes. Scores are softmax outputs, not calibrated confidence.
- Benchmarks use synthetic random data and single runs. Results will differ on other hardware, models and traffic.
- The model download in the init container fetches from GitHub at start-up. Pin or mirror the model for repeatable deployments.
- The app image reference in `kubernetes-app.yaml` points to a personal registry. Replace it with your own.

## Rollback

```bash
kubectl -n default delete -k .
```

This removes the Deployments, Services, ConfigMap, PodMonitor and the benchmark Job defined here. Review `kubectl -n default get all` first if the namespace holds other workloads.
