# AI Video Upscale Pipeline

DV/VHS interlaced video upscaling: Deinterlace (QTGMC) + Denoise (DPIR) + 2x Upscale (VHS2HD-RealPLKSR).

## Key Files

| File | Role |
|------|------|
| `upscale.sh` | Entry point: `./upscale.sh input.avi [output.mkv]` |
| `upscale.vpy` | VapourSynth script: FFmpeg decode -> QTGMC -> DPIR -> VHS2HD -> output |
| `y4m_pipe.py` | Runs upscale.vpy and writes Y4M to stdout (replaces broken vspipe) |
| `drunet_color.pth` | DPIR denoise model (DRUNet, 125MB) |
| `2xVHS2HD-RealPLKSR.pth` | 2x upscale model (RealPLKSR, 29MB) |

## Docker Image

Base: `styler00dollar/vsgan_tensorrt`. Use `:latest_no_avx512` for CPUs without AVX-512 (most Runpod instances).

## Environment

- Python packages split across two paths: `site-packages` (VS ecosystem) and `dist-packages` (PyTorch). Both are needed.
- PYTHONPATH must include `/usr/local/lib/python3.12/site-packages` for VS access.
- `trt=False` in vsspandrel calls - TensorRT Builder often fails due to CUDA version mismatch between torch_tensorrt and the host driver.

## Known Issues and Workarounds

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| Error 804: forward compatibility | CUDA compat lib overrides real driver | Disable `/etc/ld.so.conf.d/00-compat-*.conf`, run `ldconfig` |
| Illegal instruction (core dumped) | Binaries compiled for AVX-512 on non-AVX512 CPU | Use `_no_avx512` Docker tag, or rebuild FFmpeg from source |
| Failed to initialize VSScript | Embedded Python in vspipe can't find VS module | Use `y4m_pipe.py` instead of vspipe |
| No attribute resize2 | VapourSynth R73 lacks resize2 | Build `vapoursynth-resize2` plugin from source |
| DRUNet 4ch/5ch mismatch | Spandrel's `call_fn` already handles noise map | Do NOT wrap DRUNet model. Use `trt=False` |
| TRT Builder nullptr | torch_tensorrt CUDA version > driver CUDA | Use `trt=False` in vsspandrel calls |
| set_mempolicy errors | Docker seccomp policy | `docker run --cap-add SYS_NICE` |

## Setup

Two approaches:
- `docs/setup.md` - Fix up a `styler00dollar/vsgan_tensorrt` container (many workarounds)
- `docs/custom-image.md` - Build clean image from `nvidia/cuda` with VapourSynth R74 (recommended)
- `docs/troubleshooting.md` - Detailed error diagnosis
