# Implementation Findings

Research and validation findings captured during pipeline development.

## Critical API Corrections (Validated April 2026)

### vsspandrel
- **Wrong:** `from vsspandrel import inference` / `inference(clip, model_path=..., trt=False)`
- **Correct:** `from vsspandrel import vsspandrel` / `vsspandrel(clip, model_path=..., trt=False)`
- Input must be RGBH or RGBS format
- `trt=False` uses PyTorch native inference (no TensorRT)
- Source: github.com/TNTwise/vs-spandrel
- **CRITICAL:** Package is NOT on PyPI. Must install via `pip install git+https://github.com/TNTwise/vs-spandrel.git`

### vsjetpack QTGMC (QTempGaussMC)
- **Wrong:** `from vsdeinterlace import QTGMC` / `QTGMC(clip, Preset="Slow", TFF=False, ...)`
- **Correct:** `from vsdeinterlace import QTempGaussMC` / `QTempGaussMC(**kwargs).bob(clip, tff=False)`
- Uses builder pattern with snake_case parameters
- No preset system (old "Slow"/"Fast" presets don't exist)
- Parameter mapping from havsfunc:

| havsfunc | vsjetpack |
|----------|-----------|
| `Preset="Slow"` | No equivalent; defaults are good |
| `TFF=False` | `.bob(clip, tff=False)` |
| `Sharpness=0.0` | `sharpen_strength=0.0` |
| `EZDenoise=3.0` | `denoise_func=DFTTest(sigma=3.0)` |
| `DenoiseMC=True` | `denoise_mc_denoise=True` (default) |
| `GrainRestore=0.3` | `basic_noise_restore=0.3` |
| `ChromaNoise=True` | Implicit in DFTTest |
| `ChromaMotion=True` | Default in MVToolsPreset.HQ_SAD |

### vsscunet
- **Wrong:** `scunet(clip, model=0)`
- **Correct:** `from vsscunet import scunet, SCUNetModel` / `scunet(clip, model=SCUNetModel.scunet_color_real_psnr)`
- Input must be RGBH or RGBS
- `SCUNetModel` is an IntEnum (int values work but enum is canonical)
- Values: scunet_color_15=0, scunet_color_25=1, scunet_color_50=2, scunet_color_real_psnr=3, scunet_color_real_gan=4

## CUDA Compatibility

| CUDA Toolkit | PyTorch Index | Status |
|-------------|---------------|--------|
| 12.4 | cu124 | DEPRECATED - PyTorch 2.7+ dropped support |
| 12.6 | cu126 | Supported |
| 12.8 | cu128 | Supported |
| 13.0 | cu130 | Supported - latest wheels |
| 13.2 | cu130 | Forward compatible with cu130 wheels |

**Decision:** `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04` base with `cu128` PyTorch index. CUDA 13.x requires driver R580+ which Runpod does not yet deploy (still on R570/570.195.03).

## VapourSynth R74 Plugin Paths

R74 changed plugin loading. Plugins now live in the VapourSynth Python package directory:
```
$(python3 -c "import vapoursynth, os; print(os.path.join(os.path.dirname(vapoursynth.__file__), 'plugins'))")
```

Plugins installed via `meson install` go to `/usr/local/lib/vapoursynth/` — these must be symlinked to the R74 plugin directory. The `vs-link-plugins` helper script handles this.

## Source Loading Strategy

The FFmpeg subprocess pipe approach (BlankClip + ModifyFrame) is **NOT thread-safe** for VapourSynth's multi-threaded frame requests. Frames must be read sequentially from the pipe, but VS requests frames in parallel.

**Correct approach:** Use a proper source plugin with random access:
1. **lsmas** (L-SMASH-Works) - best for container formats
2. **bestsource** - Python-installable alternative
3. **ffms2** - widely available, apt-installable on Ubuntu

## SCUNet Luma-Only Strategy

SCUNet `scunet_color_real_psnr` processes all 3 RGB channels. To apply it as "luma denoise" while preserving dedicated chroma denoise work:

1. Save pre-SCUNet clip (with denoised chroma from step 5)
2. Convert to RGBS, run SCUNet
3. Convert both to YUV
4. `ShufflePlanes([scunet_out, pre_scunet, pre_scunet], [0,1,2], YUV)`
5. Result: luma from SCUNet, chroma from pre-SCUNet

## Broadwell Compatibility

Runpod Community Cloud may use Broadwell CPUs (AVX2, no AVX-512). To ensure compatibility:
- Set `CFLAGS="-march=x86-64-v2 -mtune=generic"` in Dockerfile
- FFmpeg with `--enable-runtime-cpudetect` auto-selects SIMD at runtime
- znedi3 with `make X86=1` builds SSE2/AVX2 paths
- Never compile with `-march=native` inside Docker (may target the build machine's CPU)

## Model Availability

| Model | Source | Status | Size |
|-------|--------|--------|------|
| scunet_color_real_psnr.pth | github.com/cszn/KAIR/releases | CONFIRMED | ~15MB |
| 1x-BleedOut-Compact.pth | OpenModelDB Oracle Cloud | CONFIRMED | ~4.8MB |
| 2xLiveActionV1_SPAN.pth | github.com/jcj83429/upscaling | CONFIRMED (personal repo) | ~? |
| 2xVHS2HD-RealPLKSR.pth | User-provided | NOT on OpenModelDB | ~29MB |

## Phase 1 Fixes Applied (19 items)

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | CRITICAL | CUDA 13.2 needs driver 595+, Runpod has 570 | Downgraded to CUDA 12.8.0 with cu128 |
| 2 | CRITICAL | KNLMeansCL used cmake, repo uses meson | Changed to meson build |
| 3 | FALSE POSITIVE | `fmtc.resample planes=[2,3,3]` was CORRECT | Emulator incorrectly claimed 2=fill, 3=garbage. Actual fmtconv convention: 0=fill, 1=garbage, 2=copy, 3=process. Reverted to `[2,3,3]` after p2-pipeline caught the regression. |
| 4 | CRITICAL | Missing DFTTest native plugin | Added VapourSynth-DFTTest to Dockerfile |
| 5 | CRITICAL | `from vsspandrel import inference` wrong | Fixed to `from vsspandrel import vsspandrel` |
| 6 | CRITICAL | `QTGMC()` function doesn't exist in vsjetpack | Fixed to `QTempGaussMC().bob()` with correct params |
| 7 | CRITICAL | `scunet(clip, model=0)` wrong model | Fixed to `SCUNetModel.scunet_color_real_psnr` |
| 8 | CRITICAL | `vs.get_output()` wrong R74 API | Fixed to `vs.get_output(0).clip` |
| 9 | HIGH | `-movflags +faststart` invalid for MKV | Removed |
| 10 | HIGH | `-tag:v hvc1` meaningless for MKV | Removed |
| 11 | HIGH | BleedOut on interlaced content (RGB roundtrip) | Moved after QTGMC deinterlace |
| 12 | HIGH | `set -e` prevents error messages | Fixed with `set +e` and `PIPESTATUS` |
| 13 | HIGH | `vspipe --info` loads entire pipeline as pre-check | Changed to `command -v vspipe` only |
| 14 | HIGH | vs-nlm-cuda missing SM90 for H100 | Added `-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"` |
| 15 | MODERATE | R74 plugin paths differ from meson install | Added `VAPOURSYNTH_EXTRA_PLUGIN_PATH` |
| 16 | MODERATE | No Broadwell CPU safeguard | Added `CFLAGS="-march=x86-64-v2"` |
| 17 | MODERATE | No source plugins (lsmas/bestsource/ffms2) | Added bestsource pip + ffms2 apt |
| 18 | MODERATE | CRF 18 too conservative for VHS source | Changed default to CRF 20 |
| 19 | MODERATE | SAR not in HEVC bitstream | Added `sar=16:15` to x265-params |

## Python Version Decision

Python 3.12 (Ubuntu 24.04 default) is used. Alternatives considered:
- **Python 3.13**: Supported by PyTorch 2.11 and VapourSynth R74. Has experimental JIT compiler that could speed up QTGMC's CPU-bound Python code. Risk: less battle-tested with VS plugin ecosystem.
- **Python 3.14**: Supported but `torch.compile` support is experimental. Higher risk for no clear benefit.
- **Decision**: Stick with 3.12 for v1. Zero install overhead (Ubuntu default), maximum ecosystem compatibility. Revisit 3.13 if QTGMC CPU performance needs improvement.

## vsspandrel Installation

vsspandrel (TNTwise/vs-spandrel) is **NOT on PyPI** despite the README claiming `pip install vsspandrel`. Must install from GitHub:
```
pip install git+https://github.com/TNTwise/vs-spandrel.git
```

## Blackwell GPU Support (April 2026)

**Key finding:** CUDA 12.8 already supports SM120 (Blackwell consumer/workstation). No separate build needed.

| GPU | Architecture | SM Code | VRAM | Compute |
|-----|-------------|---------|------|---------|
| RTX 5090 | Blackwell | SM120 (CC 12.0) | 32 GB GDDR7 | ~105 TFLOPS FP32 |
| RTX Pro 6000 | Blackwell | SM120 (CC 12.0) | 96 GB GDDR7 | ~115 TFLOPS FP32 |
| B100/B200 (datacenter) | Blackwell | SM100 (CC 10.0) | varies | varies |

- **SM120 ≠ SM100**: Consumer Blackwell uses SM120, datacenter uses SM100. Our pipeline targets SM120.
- **One unified build**: CUDA 12.8 + cu128 + SM120 in CUDA_ARCHS works for ALL GPUs (A100 through RTX 5090).
- **VRAM budget**: Peak ~2.2GB on RTX 5090's 32GB — massive headroom.
- **Performance**: RTX 5090 comparable to H100 PCIe for our small-batch inference workload.
- **FP8**: Blackwell supports FP8 natively but our models are small enough that FP16 is the practical sweet spot.
- **Multi-GPU**: 2x RTX 5090 via `CUDA_VISIBLE_DEVICES` isolation, each processing one video independently.

## Known Risks

1. **vs-nlm-cuda may not build** — KNLMeansCL OpenCL fallback included, plus graceful skip
2. **SPAN model URL is from a personal repo** — URL may break; consider mirroring
3. **VapourSynth R74 is very new** (April 2026) — some plugins may have compatibility issues
4. **No checkpoint/resume** — pipeline crash means full reprocess
5. **CAS plugin may not build** — graceful skip with warning
6. **CUDA version**: Using 12.8 (driver 570+ required). Runpod runs 570.195.03. CUDA 13.x needs driver 580+ which Runpod does not yet deploy
