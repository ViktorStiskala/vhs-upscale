"""
VHS Restoration Pipeline - VapourSynth Script
10-step pipeline: decode -> deinterlace -> chroma fix -> denoise -> 4x upscale -> encode

Usage:
    vspipe --y4m upscale.vpy - | ffmpeg -i pipe:0 ...
    or via upscale.sh entry point

Environment variables:
    INPUT_FILE       - Path to source AVI file (required)
    MODEL_DIR        - Directory containing built-in .pth models (default: /models)
    USER_MODEL_DIR   - Directory for user-provided models (default: /workspace/models)
    CHROMA_SHIFT_SX  - Chroma shift correction in pixels (default: 0, typical PAL VHS: -2 to -6)
    CAS_STRENGTH     - CAS sharpening strength 0.0-1.0 (default: 0.4)
    DENOISE_STRENGTH - NLM chroma denoise strength (default: 3.0)
    NUM_THREADS      - VapourSynth thread count (default: auto)
    SKIP_UPSCALE2    - Set to "1" to skip second upscale (output 1440x1152 instead of 2880x2304)
    FIELD_ORDER      - Field order: auto, bff, tff, progressive (default: auto, detects from source)
"""

import os
import sys
import vapoursynth as vs

core = vs.core

# ============================================================
# Configuration
# ============================================================
INPUT_FILE = os.environ.get("INPUT_FILE", "")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
USER_MODEL_DIR = os.environ.get("USER_MODEL_DIR", "/workspace/models")
CHROMA_SHIFT_SX = float(os.environ.get("CHROMA_SHIFT_SX", "0"))
CAS_STRENGTH = float(os.environ.get("CAS_STRENGTH", "0.4"))
DENOISE_STRENGTH = float(os.environ.get("DENOISE_STRENGTH", "3.0"))
NUM_THREADS = int(os.environ.get("NUM_THREADS", "0"))
SKIP_UPSCALE2 = os.environ.get("SKIP_UPSCALE2", "0") == "1"
# Field order: "bff" (PAL DV default), "tff", or "progressive" (skip QTGMC)
FIELD_ORDER = os.environ.get("FIELD_ORDER", "auto").lower()

if NUM_THREADS > 0:
    core.num_threads = NUM_THREADS

if not INPUT_FILE:
    raise ValueError("INPUT_FILE environment variable must be set")

# ============================================================
# Model paths
# ============================================================
BLEEDOUT_MODEL = os.path.join(MODEL_DIR, "1x-BleedOut-Compact.pth")
SCUNET_MODEL = os.path.join(MODEL_DIR, "scunet_color_real_psnr.pth")
SPAN_MODEL = os.path.join(MODEL_DIR, "2xLiveActionV1_SPAN.pth")
VHS2HD_MODEL = os.path.join(USER_MODEL_DIR, "2xVHS2HD-RealPLKSR.pth")

for name, path in [("BleedOut", BLEEDOUT_MODEL), ("SCUNet", SCUNET_MODEL),
                    ("VHS2HD", VHS2HD_MODEL)]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} model not found: {path}")
if not SKIP_UPSCALE2 and not os.path.exists(SPAN_MODEL):
    raise FileNotFoundError(f"SPAN model not found: {SPAN_MODEL}")

# ============================================================
# Import AI inference
# ============================================================
from vsspandrel import vsspandrel as run_model

# ============================================================
# Step 1: Source decoding
# ============================================================
# Use proper source plugins with random access (thread-safe).
# FFmpeg subprocess pipe is NOT used — it's not thread-safe for VS.
clip = None

for loader in [
    lambda: core.lsmas.LWLibavSource(INPUT_FILE),
    lambda: core.bs.VideoSource(INPUT_FILE),
    lambda: core.ffms2.Source(INPUT_FILE),
]:
    if clip is not None:
        break
    try:
        clip = loader()
    except (AttributeError, vs.Error):
        pass

if clip is None:
    raise RuntimeError(
        "No source plugin available. Install lsmas, bestsource, or ffms2."
    )

# Verify source format (PAL DV should decode as YUV420P8)
if clip.format.id != vs.YUV420P8:
    sys.stderr.write(
        f"WARNING: Source decoded as {clip.format.name}, expected YUV420P8. "
        f"Pipeline may produce unexpected results for non-standard DV formats.\n"
    )

# Detect or set field order
if FIELD_ORDER == "auto":
    # Try to detect from source frame properties
    try:
        props = clip.get_frame(0).props
        fb = props.get("_FieldBased", 0)
        if fb == 1:
            FIELD_ORDER = "bff"
        elif fb == 2:
            FIELD_ORDER = "tff"
        elif fb == 0:
            FIELD_ORDER = "progressive"
        else:
            FIELD_ORDER = "bff"  # Default PAL DV assumption
    except Exception:
        FIELD_ORDER = "bff"  # Default PAL DV assumption
    sys.stderr.write(f"Field order detected/assumed: {FIELD_ORDER}\n")

if FIELD_ORDER == "progressive":
    sys.stderr.write("Source is progressive — skipping QTGMC deinterlace.\n")
elif FIELD_ORDER in ("bff", "tff"):
    fb_value = 1 if FIELD_ORDER == "bff" else 2
    clip = core.std.SetFieldBased(clip, fb_value)
else:
    raise ValueError(f"Invalid FIELD_ORDER: {FIELD_ORDER}. Use 'auto', 'bff', 'tff', or 'progressive'.")

# ============================================================
# Step 2: Chroma shift correction (on interlaced, before deinterlace)
# ============================================================
# PAL VHS typically has chroma shifted left by 2-6 pixels.
# This operates on UV planes only — safe on interlaced content.
if CHROMA_SHIFT_SX != 0:
    try:
        # fmtconv planes convention: 0=fill constant, 1=garbage, 2=copy, 3=process
        # Copy luma unchanged (2), process/shift both chroma planes (3)
        clip = core.fmtc.resample(clip, sx=CHROMA_SHIFT_SX, planes=[2, 3, 3])
    except AttributeError:
        sys.stderr.write("WARNING: fmtconv not available, skipping chroma shift\n")

# ============================================================
# Step 3: Deinterlace (QTGMC bob -> 50fps progressive)
# ============================================================
# Run QTGMC BEFORE AI models. AI models (BleedOut, SCUNet, upscalers) are
# trained on progressive content. Running them on interlaced frames with
# combed artifacts produces suboptimal results.
if FIELD_ORDER in ("bff", "tff"):
    from vsdeinterlace import QTempGaussMC
    from vsdenoise import DFTTest

    tff = (FIELD_ORDER == "tff")
    clip = QTempGaussMC(
        sharpen_strength=0.0,              # Disable — CAS applied after upscale
        denoise_func=DFTTest(sigma=3.0),   # Temporal denoise (replaces EZDenoise=3.0)
        denoise_mc_denoise=True,           # Motion-compensated denoise
        basic_noise_restore=0.3,           # Light grain to prevent plastic look
    ).bob(clip, tff=tff)
    # Result: 25fps interlaced -> 50fps progressive, ~278K -> ~556K frames
else:
    sys.stderr.write("Progressive source — QTGMC skipped, framerate unchanged.\n")

# ============================================================
# Step 4: Chroma bleed fix (1x BleedOut Compact)
# ============================================================
# Now on progressive frames — safe for RGB roundtrip.
# BleedOut is a 1x model (no scaling) that fixes color bleed and VHS rainbows.
clip_rgb = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="470bg")
clip_rgb = run_model(clip_rgb, model_path=BLEEDOUT_MODEL, trt=False)
clip = core.resize.Bicubic(clip_rgb, format=vs.YUV420P8, matrix_s="470bg")
del clip_rgb

# ============================================================
# Step 5: Chroma denoise (aggressive, GPU-accelerated)
# ============================================================
# VHS chroma bandwidth is ~0.6 MHz (vs ~3 MHz luma).
# Aggressive chroma denoise removes noise without losing real detail.
chroma_denoised = False

if not chroma_denoised:
    try:
        clip = core.nlm_cuda.NLMeans(
            clip, a=2, h=DENOISE_STRENGTH, d=1, s=4, channels="UV"
        )
        chroma_denoised = True
    except (AttributeError, vs.Error):
        pass

if not chroma_denoised:
    try:
        clip = core.knlm.KNLMeansCL(
            clip, a=2, h=DENOISE_STRENGTH, d=3,
            channels="UV", device_type="gpu"
        )
        chroma_denoised = True
    except (AttributeError, vs.Error):
        pass

if not chroma_denoised:
    sys.stderr.write("WARNING: No NLM denoiser available, skipping chroma denoise\n")

# ============================================================
# Step 6: Luma denoise (SCUNet — luma-only merge)
# ============================================================
# SCUNet scunet_color_real_psnr processes all RGB channels.
# To preserve dedicated chroma denoise from step 5:
#   1. Run SCUNet on full RGB
#   2. Convert both to YUV
#   3. Take Y from SCUNet, UV from pre-SCUNet
clip_pre_scunet = clip

clip_rgb = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="470bg")

try:
    from vsscunet import scunet, SCUNetModel
    clip_rgb = scunet(clip_rgb, model=SCUNetModel.scunet_color_real_psnr, trt=False)
except (ImportError, Exception):
    clip_rgb = run_model(clip_rgb, model_path=SCUNET_MODEL, trt=False)

clip_denoised = core.resize.Bicubic(clip_rgb, format=vs.YUV420P8, matrix_s="470bg")
del clip_rgb

# Luma-only merge: Y from SCUNet, UV from pre-SCUNet
clip = core.std.ShufflePlanes(
    [clip_denoised, clip_pre_scunet, clip_pre_scunet],
    planes=[0, 1, 2],
    colorfamily=vs.YUV
)
del clip_denoised, clip_pre_scunet

# ============================================================
# Step 7: Upscale pass 1 — 2x VHS2HD-RealPLKSR (720x576 -> 1440x1152)
# ============================================================
clip_rgb = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="470bg")
clip_rgb = core.resize.Point(clip_rgb, format=vs.RGBH)  # FP16 for speed
clip_rgb = run_model(clip_rgb, model_path=VHS2HD_MODEL, trt=False)

# ============================================================
# Step 8: Upscale pass 2 — 2x SPAN (1440x1152 -> 2880x2304)
# ============================================================
if not SKIP_UPSCALE2:
    clip_rgb = run_model(clip_rgb, model_path=SPAN_MODEL, trt=False)

# ============================================================
# Step 9: CAS sharpen + BT.601 -> BT.709 color space conversion
# ============================================================
# Convert RGBH -> RGBS -> YUV420P10 BT.709 in one step.
# This handles the color matrix conversion from BT.601 (470bg) working space
# to BT.709 for HD output.
clip_rgb = core.resize.Point(clip_rgb, format=vs.RGBS)
clip = core.resize.Bicubic(clip_rgb, format=vs.YUV420P10, matrix_s="709")
del clip_rgb

if CAS_STRENGTH > 0:
    try:
        clip = core.cas.CAS(clip, sharpness=CAS_STRENGTH, planes=[0])
    except AttributeError:
        sys.stderr.write("WARNING: CAS plugin not available, skipping sharpening\n")

# SAR 16:15 preserved via FFmpeg output metadata (-aspect 4:3 + sar=16:15)

# ============================================================
# Output
# ============================================================
clip.set_output()
