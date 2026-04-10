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

## Dockerfile Gotchas

- vsjetpack has undocumented hard dependencies on native plugins: `resize2`, `akarin`, `noise` — all must be built and verified
- resize2's meson.build installs to Python site-packages (not --libdir); needs explicit copy to /opt/vs-plugins/
- akarin plugin requires LLVM >= 10, < 16 (`llvm-15-dev` + `libzstd-dev` on Ubuntu 24.04)
- vapoursynth.pc in stage-plugins must use release version (74), not API version (4) — vs-noise checks `>=55`
- Never use `|| echo "... failed, skipping"` for plugins that vsjetpack requires — the error surfaces at runtime, not build time
- Build-time verification (end of Dockerfile) must assert all required plugin namespaces
- Test plugin builds locally: `docker run --rm nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 bash -c "..."`

## Dependency Verification

Python wrapper packages (vsjetpack, vsscunet, vs-spandrel) are unpinned and their APIs
can change without warning. The Dockerfile build-time verification only checks imports —
it does NOT verify that specific function kwargs still exist. A kwarg rename upstream
means a successful Docker build that fails at runtime.

Run before Docker builds:

```bash
python3 verify_deps.py          # clone repos to .tmp/, check API signatures + paths
python3 verify_deps.py --clean  # re-clone from scratch
```

The script:
1. Clones upstream source of vs-spandrel, vs-jetpack, vs-scunet into `.tmp/`
2. Parses their Python source with `ast` to extract function/class signatures
3. Checks that every kwarg used in `upscale.vpy` still exists upstream
4. Cross-references path defaults between `upscale.vpy`, `upscale.sh`, and `Dockerfile`

### When to run

- Before every `docker build` (5 seconds vs 10+ min wasted build)
- After editing `upscale.vpy` function calls or kwargs
- After changing pip install lines in Dockerfile
- Periodically to catch upstream breakage early

### Maintaining the manifest

When adding new API calls to `upscale.vpy`, add matching checks to the `REPOS` dict
in `verify_deps.py`. Each check specifies: function/class name, required params, and
a description linking to the `upscale.vpy` line number.

### Path consistency rules

Path defaults must match across three files:
- `upscale.vpy`: `os.environ.get("MODEL_DIR", "/models")`
- `upscale.sh`: `MODEL_DIR="${MODEL_DIR:-/models}"`
- `Dockerfile`: `ADD ... /models/`, `mkdir -p /models`

If changing any path in Dockerfile (models, plugins, scripts), update the matching
defaults in `upscale.vpy` and `upscale.sh`, then run `verify_deps.py`.

### Unpinned packages (fragile surface)

| Package | Provides | Risk |
|---------|----------|------|
| `vsjetpack` (PyPI) | `vsdeinterlace.QTempGaussMC`, `vsdenoise.DFTTest` | High — complex API with many kwargs |
| `vsscunet` (PyPI) | `scunet`, `SCUNetModel` | Medium — enum members could change |
| `vs-spandrel` (git) | `vsspandrel` | Medium — called 4x with model_path + trt |
| `spandrel` (PyPI) | Internal dep of vs-spandrel | Low — not called directly |

Native C plugins (mvtools, fmtconv, znedi3, etc.) use the stable VapourSynth C API
and are verified by the Dockerfile build-time assertions. They are NOT checked by
`verify_deps.py`.
