# AI Video Upscale Pipeline

Upscale interlaced SD video (DV/VHS, 720x576i) to progressive HD (1440x1152p) using AI models on GPU.

## Pipeline

```
Input AVI (720x576i interlaced)
  |
  v
FFmpeg decode (raw YUV420P via pipe)
  |
  v
QTGMC deinterlace (CPU, vsjetpack QTempGaussMC)
  720x576i BFF -> 720x576p @ 50fps
  |
  v
DPIR denoise (GPU, DRUNet via vsspandrel, FP32)
  720x576p -> 720x576p denoised
  |
  v
VHS2HD 2x upscale (GPU, RealPLKSR via vsspandrel, FP16)
  720x576p -> 1440x1152p
  |
  v
FFmpeg encode (libx265 CRF 18, FLAC audio)
  |
  v
Output MKV (1440x1152p, YUV420P10, HEVC)
```

## Quick Start

```bash
./upscale.sh input.avi                   # output: input_upscaled.mkv
./upscale.sh input.avi custom_name.mkv   # explicit output path
```

## Docker Image Options

### Option A: Pre-built `styler00dollar/vsgan_tensorrt` (quick start)

Has VapourSynth R73, PyTorch, TensorRT, and many plugins pre-installed. Requires workarounds (see [setup.md](setup.md)).

| Tag | CPU Requirement | TensorRT | Size |
|-----|----------------|----------|------|
| `latest` | AVX-512 (Xeon, EPYC 9xx4) | 10.15 | ~40GB |
| `latest_no_avx512` | Any x86_64 (EPYC 7xx3, consumer) | 10.13 | ~28GB |

Check your CPU: `grep -c avx512 /proc/cpuinfo` (0 = use `_no_avx512`).

### Option B: Custom image from `nvidia/cuda` (recommended for clean setup)

Build a clean image with VapourSynth R74 (pip install), matching CUDA, no compat lib issues. See [custom-image.md](custom-image.md) for the complete Dockerfile.

Base: `nvidia/cuda:XX.X.X-cudnn-devel-ubuntu24.04` (match your GPU driver's CUDA version).

**Advantages over Option A**: VapourSynth R74 (resize2 built-in, no plugin build needed), CUDA version matches driver (no compat lib workaround), FFmpeg built for your CPU (no Illegal instruction), ~10-15GB smaller.

### Runpod

Add `--cap-add SYS_NICE` to the docker command for optimal x265 NUMA performance.

## Required Models

| Model | Size | Purpose | Source |
|-------|------|---------|--------|
| `drunet_color.pth` | 125MB | DPIR denoising | https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth |
| `2xVHS2HD-RealPLKSR.pth` | 29MB | 2x upscaling | User-provided |

## Documentation

- [Setup Guide](setup.md) - Environment setup for `styler00dollar/vsgan_tensorrt` containers
- [Custom Image](custom-image.md) - Build your own image from `nvidia/cuda` (recommended)
- [Architecture](architecture.md) - Technical details and design decisions
- [Troubleshooting](troubleshooting.md) - Common errors and fixes
