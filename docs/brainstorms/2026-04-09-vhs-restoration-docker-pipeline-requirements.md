---
date: 2026-04-09
topic: vhs-restoration-docker-pipeline
---

# VHS Restoration Docker Pipeline for Runpod

## Problem Frame

We have ~3 hours of PAL DV-captured VHS footage (720x576i, 25fps, DV codec, ~40GB AVI) that needs AI-powered restoration and 4x upscaling. The existing research (`projects/vhs-restoration/RESEARCH.md`) and prototype docs (`projects/vhs-restoration/research/docs/`) define the pipeline but no working Dockerfile or scripts exist in the repo yet. We need a reproducible Docker image deployable on Runpod H100 PCIe 80GB that runs the full pipeline end-to-end.

## Pipeline Flow

```
AVI (720x576i PAL DV, BFF)
  |
  | 1. FFmpeg decode (or lsmas if R74 vspipe works)
  v
  | 2. Chroma shift correction (fmtc.resample sx)
  | 3. Chroma bleed fix (1x BleedOut Compact)
  v
  | 4. Deinterlace (QTGMC "Slow", bob to 50fps)
  v
  | 5. Chroma denoise (vs-nlm-cuda on UV, aggressive)
  | 6. Luma denoise (SCUNet scunet_color_real_psnr)
  v
  | 7. Upscale pass 1: 2xVHS2HD-RealPLKSR (720x576 -> 1440x1152)
  | 8. Upscale pass 2: 2x SPAN general model (1440x1152 -> 2880x2304)
  v
  | 9. CAS sharpen on luma (0.3-0.5)
  | 10. BT.601 -> BT.709 color space conversion
  v
  FFmpeg encode (libx265 + FLAC audio) -> MKV (2880x2304p, 50fps)
```

## Requirements

**Docker Image**

- R1. Dockerfile based on `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04` with parameterized CUDA version via build args (cu126/cu128/cu130).
- R2. PyTorch 2.11.0 with matching CUDA index, VapourSynth R74 via pip, vsjetpack 1.3.1 for QTGMC.
- R3. All required native VapourSynth plugins built from source: mvtools, fmtconv, znedi3, resize2, CAS, vs-nlm-cuda.
- R4. FFmpeg built from source with libx264, libx265, libsvtav1, NVENC/NVDEC, and runtime CPU detection.
- R5. AI models downloaded at build time: SCUNet (scunet_color_real_psnr), 1x BleedOut Compact, 2x SPAN (2xNomosUni_span_multijpg or 2xLiveActionV1_SPAN). 2xVHS2HD-RealPLKSR provided by user at runtime via /workspace mount.
- R6. Image pushed to a container registry for use as a Runpod template.

**Pipeline Scripts**

- R7. VapourSynth script (`upscale.vpy`) implementing the full 10-step pipeline: chroma shift correction, chroma bleed fix (BleedOut), QTGMC deinterlace, chroma denoise (vs-nlm-cuda), luma denoise (SCUNet), 2x VHS2HD upscale, 2x SPAN upscale, CAS sharpen, BT.601->BT.709 conversion.
- R8. Shell entry point (`upscale.sh`) that orchestrates: VapourSynth processing piped to FFmpeg encoding with audio muxing from the original file.
- R9. Pipeline must handle the full source (~278K input frames, ~556K output frames after bob deinterlace, ~3hr) without manual intervention.

**Runpod Deployment**

- R10. All persistent data (source videos, output files, user-provided models) stored in `/workspace` (Runpod network volume, minimum 300GB to accommodate ~40GB source, ~150-200GB output, and models). Everything else on the container filesystem is ephemeral. Intermediate processing artifacts may use ephemeral storage.
- R11. On-demand pod (not spot) for long-running jobs. H100 PCIe 80GB ($1.99/hr) with 16 vCPUs.

**Quality**

- R12. 4x output via two-stage chain: 2xVHS2HD-RealPLKSR (VHS-specific) then 2x SPAN (general-purpose). Final output: 2880x2304p, 50fps, YUV420P10, HEVC.
- R13. Separate chroma/luma processing paths as recommended in RESEARCH.md: aggressive chroma denoise, moderate luma denoise.

## Success Criteria

- Docker image builds successfully and runs the full pipeline on an H100 PCIe 80GB Runpod pod
- Source AVI (720x576i) produces a 2880x2304p 50fps HEVC MKV with visibly improved quality (reduced noise, no chroma bleed, sharp detail)
- Pipeline processes the full ~3hr source end-to-end without OOM or crashes
- All models and scripts survive pod stop/restart via /workspace persistence

## Scope Boundaries

- **Not in scope:** TensorRT optimization (defer to later iteration -- use PyTorch native inference with `trt=False` first)
- **Not in scope:** vs-mfdin GPU deinterlacing (experimental; use proven QTGMC for now)
- **Not in scope:** BM3DCUDA temporal denoising (complex build; SCUNet single-frame is the first iteration)
- **Not in scope:** TAPE model integration (standalone PyTorch pipeline, no VapourSynth wrapper)
- **Not in scope:** Automated scene detection or batch processing of multiple files

## Key Decisions

- **CUDA 12.8 base (not 12.4.1):** PyTorch 2.7+ dropped cu124 support. CUDA 12.8 is the sweet spot for current ecosystem compatibility.
- **SCUNet replaces DPIR/DRUNet:** Better noise model for VHS's complex artifacts. The `scunet_color_real_psnr` variant avoids color shifts.
- **vs-nlm-cuda replaces KNLMeansCL:** CUDA-native, no OpenCL dependency needed in Docker. Drop-in replacement for chroma denoising.
- **2xLiveActionV1_SPAN as second upscaler:** Available directly from GitHub (no Google Drive dependency), trained on live-action video frames with compression removal and dehalo. Alternative: 2xNomosUni_span_multijpg if testing shows better results.
- **50fps bob output (not 25fps):** QTGMC bob produces both fields as progressive frames, preserving temporal detail from interlaced source.
- **PyTorch native inference (no TRT):** Avoids CUDA version mismatch issues with torch_tensorrt. TRT optimization is a future iteration.

## Dependencies / Assumptions

- User has the 2xVHS2HD-RealPLKSR.pth model file (not available on OpenModelDB as of April 2026)
- Runpod H100 PCIe 80GB pods run NVIDIA driver 550+ supporting CUDA 12.8
- Source files are DV-captured VHS (PAL, BFF interlaced) in AVI container

## Outstanding Questions

### Deferred to Planning

- [Affects R7][Technical] Exact chroma shift `sx` value for this specific VHS deck/capture chain -- may need per-source tuning, consider making it a parameter
- [Affects R7][Needs research] Whether VapourSynth R74's `vspipe` works correctly (eliminating need for `y4m_pipe.py` workaround)
- [Affects R4][Needs research] Whether `lsmas` or `bestsource` source filters work in R74, eliminating the FFmpeg decode pipe
- [Affects R5][Needs research] Best 2x SPAN model for this use case -- may need A/B testing between 2xLiveActionV1_SPAN and 2xNomosUni_span_multijpg
- [Affects R8][Technical] FFmpeg encoding parameters: CRF value, preset, x265 tuning options for upscaled VHS content
- [Affects R3][Needs research] vs-nlm-cuda build requirements and compatibility with VapourSynth R74

## Validated Component Versions (April 2026)

| Component | Version | Install Method |
|-----------|---------|----------------|
| CUDA base image | 12.8.0-cudnn-devel-ubuntu24.04 | Docker FROM |
| PyTorch | 2.11.0 | pip (cu128 index) |
| VapourSynth | R74 | pip install vapoursynth |
| vsjetpack | 1.3.1 | pip (requires Python 3.12+) |
| vsspandrel | 1.1.0 | pip |
| vsscunet | 2.0.0 | pip |
| vs-mlrt | v15.16 | GitHub release (future: TRT optimization) |
| vs-rife | 5.7.0 | pip (future: frame interpolation) |

## Model Downloads

| Model | URL | Size |
|-------|-----|------|
| SCUNet (PSNR) | `https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth` | ~15MB |
| 1x BleedOut Compact | `https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/1x-BleedOut-Compact.pth` | ~4.8MB |
| 2xLiveActionV1_SPAN | `https://raw.githubusercontent.com/jcj83429/upscaling/f73a3a02874360ec6ced18f8bdd8e43b5d7bba57/2xLiveActionV1_SPAN/2xLiveActionV1_SPAN_490000.pth` | ~? |
| 2xVHS2HD-RealPLKSR | User-provided via /workspace | ~29MB |

## Next Steps

`-> /ce:plan` for structured implementation planning
