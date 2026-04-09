# Restoration Pipeline

The VapourSynth pipeline (`upscale.vpy`) performs 10 processing steps on each frame. The ordering is deliberate -- deinterlacing must happen before AI models (trained on progressive content), and chroma operations are sequenced to avoid double-processing.

## Step 1: Source Decode

Loads the AVI file using the first available VapourSynth source plugin: BestSource, FFMS2, or L-SMASH Works. These provide random-access frame serving, which is required for VapourSynth's multi-threaded architecture.

**Input:** PAL DV AVI (720x576i, YUV420P8, 25fps interlaced)

## Step 2: Chroma Shift Correction

PAL VHS recordings have chroma (color) shifted horizontally by 2-6 pixels relative to luma. This step uses `fmtconv` to shift UV planes only, correcting the misalignment. Operates on interlaced content before deinterlace -- UV planes don't have interlacing artifacts so this is safe.

**Plugin:** fmtconv (`core.fmtc.resample`)
**Parameter:** `--chroma-shift` (default: 0, typical: -3)

## Step 3: QTGMC Deinterlace

Converts interlaced 25fps to progressive 50fps using motion-compensated temporal deinterlacing (bob mode). QTGMC is the gold standard for deinterlacing -- it uses motion vectors (mvtools) and neural net interpolation (znedi3) to reconstruct full frames from fields.

This step must run before AI models because they're trained on progressive content. Running them on interlaced frames with combed artifacts produces suboptimal results.

**Plugin:** QTempGaussMC from vsjetpack (uses mvtools + znedi3 + DFTTest)
**Output:** 50fps progressive (~556K frames from a 3hr PAL source)

## Step 4: Chroma Bleed Fix (BleedOut)

Fixes VHS-specific color bleeding and rainbow artifacts using the 1x BleedOut Compact neural network. This is a 1x model (no scaling) that operates in RGB space.

**Model:** `1x-BleedOut-Compact.pth` (built into Docker image)
**Format:** YUV420P8 -> RGBS -> model -> YUV420P8

## Step 5: Chroma Denoise (NLM)

Aggressive non-local means denoising on UV (chroma) planes only. VHS chroma bandwidth is ~0.6 MHz vs ~3 MHz for luma, so heavy chroma denoising removes noise without losing real detail.

Prefers CUDA NLM (`vs-nlm-cuda`) for speed, falls back to OpenCL KNLMeansCL.

**Plugin:** nlm_cuda or KNLMeansCL
**Parameter:** `--denoise` (default: 3.0)

## Step 6: Luma Denoise (SCUNet)

SCUNet denoises all RGB channels, but we only take the luma (Y) result to preserve the dedicated chroma denoise from step 5. The UV planes from the pre-SCUNet frame are kept.

**Model:** `scunet_color_real_psnr.pth` (built into Docker image)
**Technique:** Run SCUNet on RGB, convert both to YUV, ShufflePlanes: Y from SCUNet + UV from pre-SCUNet

## Step 7: Upscale Pass 1 -- 2x VHS2HD (720x576 -> 1440x1152)

First upscale pass using a VHS-specific 2x model trained on VHS artifacts. Operates in RGBH (FP16) for GPU memory efficiency.

**Model:** `2xVHS2HD-RealPLKSR.pth` (user-provided, in `/workspace/models/`)

## Step 8: Upscale Pass 2 -- 2x SPAN (1440x1152 -> 2880x2304)

Second upscale pass using a general-purpose 2x model for clean detail enhancement. Can be skipped with `--skip-upscale2` for 2x output only.

**Model:** `2xLiveActionV1_SPAN.pth` (built into Docker image)

## Step 9: CAS Sharpen + Color Space Conversion

Applies Contrast Adaptive Sharpening to the luma plane, then converts from BT.601 (SD working space) to BT.709 (HD standard). Output is YUV420P10 for 10-bit encoding precision.

**Plugin:** CAS (`core.cas.CAS`)
**Parameter:** `--cas` (default: 0.4, range: 0.0-1.0)

## Step 10: FFmpeg Encode

The VapourSynth output is piped as Y4M to FFmpeg, which encodes video with libx265 (HEVC) and muxes audio from the original source as FLAC.

**Output:** MKV container, HEVC video (YUV420P10), FLAC audio
**Parameters:** `--crf` (default: 20), `--preset` (default: slow)
**Metadata:** SAR 16:15 preserved for correct 4:3 display aspect ratio
