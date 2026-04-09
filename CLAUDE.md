# VHS Restoration Pipeline

AI-powered VHS restoration and 4x upscaling pipeline running on Runpod GPU pods.

## Key Files

| File | Role |
|------|------|
| `Dockerfile` | Docker image: nvidia/cuda 12.8 + VapourSynth R74 + PyTorch + all plugins |
| `upscale.vpy` | VapourSynth pipeline: 10-step restoration and 4x upscale |
| `upscale.sh` | Entry point: `./upscale.sh input.avi [output.mkv]` |
| `batch_upscale.sh` | Multi-GPU batch processing: distributes videos across GPUs |
| `setup.sh` | First-run setup: verifies GPU, plugins, models, creates /workspace dirs |
| `docs/RUNPOD.md` | Runpod setup, volume config, running instructions |

## Pipeline

```
AVI (720x576i PAL DV) → Source decode (lsmas/bestsource/ffms2)
→ Chroma shift correction → QTGMC deinterlace (bob to 50fps)
→ BleedOut chroma fix → Chroma denoise (NLM on UV)
→ Luma denoise (SCUNet) → 2x VHS2HD upscale → 2x SPAN upscale
→ CAS sharpen → BT.601→709 → FFmpeg encode → MKV (2880x2304p 50fps HEVC)
```

## Source Material

- PAL DV-captured VHS: 720x576i, 25fps, BFF, SAR 16:15 (4:3 display)
- DV codec in AVI container, ~40GB per 3hr tape
- Non-square pixels: pipeline preserves SAR 16:15 in output

## Models

Models baked into Docker image (downloaded at build time):
- `scunet_color_real_psnr.pth` — SCUNet denoiser (replaces DPIR/DRUNet)
- `1x-BleedOut-Compact.pth` — Chroma bleed/rainbow fix
- `2xLiveActionV1_SPAN.pth` — General-purpose 2x upscaler (second pass)

User-provided model (must be in `/workspace/models/`):
- `2xVHS2HD-RealPLKSR.pth` — VHS-specific 2x upscaler (first pass)

## Docker

```bash
docker build -t vhs-restore .
```

One image works for ALL supported GPUs. CUDA 12.8 + cu128 supports:
- A100/H100 (SM80/SM90) — Runpod driver R570+
- RTX 5090/RTX Pro 6000 (SM120 Blackwell) — driver R570+
- SM120 in CUDA architecture targets. Custom GPU plugins (vs-nlm-cuda) compiled natively for Blackwell

Run on Runpod: see `docs/RUNPOD.md`

## Runpod Conventions

- `/workspace` is the persistent network volume (survives pod stop/restart)
- Everything else on the container is ephemeral
- Source videos, models, and outputs go in `/workspace/`
- Use on-demand pods (not spot) for long-running jobs

## Development Notes

- Use `uv` for Python dependency management per monorepo conventions
- VapourSynth R74 is pip-installable (`pip install vapoursynth`)
- QTGMC via vsjetpack (`from vsdeinterlace import QTGMC`)
- AI models via vsspandrel (single-image) or dedicated plugins (temporal)
- No TensorRT in this iteration — use PyTorch native inference (`trt=False`)
- All native VS plugins (mvtools, fmtconv, znedi3, etc.) built from source in Docker
