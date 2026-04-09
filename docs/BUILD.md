# Building the Docker Image

## Pre-built Image

A pre-built image is available on GitHub Container Registry:

```bash
docker pull ghcr.io/viktorstiskala/vhs-upscale:cu128
```

## Build from Source

The Dockerfile uses multi-stage builds so PyTorch download, FFmpeg compilation, and plugin builds run in parallel. BuildKit is required.

```bash
# Recommended: buildx (parallel stages, fastest)
docker buildx build -t ghcr.io/viktorstiskala/vhs-upscale:cu128 --load .
docker push ghcr.io/viktorstiskala/vhs-upscale:cu128

# Alternative: DOCKER_BUILDKIT=1 also enables parallel stages
DOCKER_BUILDKIT=1 docker build -t ghcr.io/viktorstiskala/vhs-upscale:cu128 .
```

### CUDA 13.0 variant

Only needed for FP4/FP8 advanced CUDA features (not used in current pipeline). Requires driver R580+.

```bash
docker buildx build \
  --build-arg CUDA_TAG=13.0.2-cudnn-devel-ubuntu24.04 \
  --build-arg CU_TAG=cu130 \
  -t ghcr.io/viktorstiskala/vhs-upscale:cu130 --load .
docker push ghcr.io/viktorstiskala/vhs-upscale:cu130
```

## Supported GPUs

CUDA 12.8 with cu128 natively supports all target GPUs including Blackwell (SM120). The image includes compiled GPU code for SM75 through SM120 -- no JIT compilation needed.

| GPU | Architecture | SM | VRAM | Driver | Status |
|-----|-------------|-----|------|--------|--------|
| A100 SXM/PCIe | Ampere | 80 | 80 GB | R570+ | Tested |
| H100 SXM/PCIe | Hopper | 90 | 80 GB | R570+ | Tested |
| RTX 5090 | Blackwell | 120 | 32 GB | R570+ | Supported |
| RTX Pro 6000 | Blackwell | 120 | 96 GB | R570+ | Supported |
| RTX 4090 | Ada | 89 | 24 GB | R570+ | Supported (24GB tight) |

## Build Architecture

The Dockerfile is structured as parallel build stages:

```
base (apt packages)
├── stage-python  (PyTorch + VapourSynth ecosystem)  ~230s
├── stage-ffmpeg  (FFmpeg from source)               ~120s
├── stage-plugins (native VS plugins + CUDA NLM)     ~60s
└── final (merge + BestSource + models + verify)
```
