#!/usr/bin/env bash
# VHS Restoration Pipeline - Entry Point
#
# Usage:
#   ./upscale.sh input.avi [output.mkv]
#   ./upscale.sh input.avi output.mkv [--chroma-shift -3] [--cas 0.5] [--crf 18]
#
# Environment variables (alternative to flags):
#   CHROMA_SHIFT_SX  - Chroma shift correction pixels (default: 0)
#   CAS_STRENGTH     - CAS sharpening 0.0-1.0 (default: 0.4)
#   DENOISE_STRENGTH - NLM chroma denoise strength (default: 3.0)
#   CRF              - x265 CRF value (default: 20)
#   PRESET           - x265 preset (default: slow)
#   NUM_THREADS      - VapourSynth threads (default: auto)
#   SKIP_UPSCALE2    - Set to "1" to skip second upscale pass (output 1440x1152)
#   FIELD_ORDER      - Field order: auto, bff, tff, progressive (default: auto)

set -euo pipefail

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPY_SCRIPT="${SCRIPT_DIR}/upscale.vpy"
MODEL_DIR="${MODEL_DIR:-/models}"
USER_MODEL_DIR="${USER_MODEL_DIR:-/workspace/models}"

# ============================================================
# Parse arguments
# ============================================================
INPUT_FILE=""
OUTPUT_FILE=""
CHROMA_SHIFT_SX="${CHROMA_SHIFT_SX:-0}"
CAS_STRENGTH="${CAS_STRENGTH:-0.4}"
DENOISE_STRENGTH="${DENOISE_STRENGTH:-3.0}"
CRF="${CRF:-20}"
PRESET="${PRESET:-slow}"
NUM_THREADS="${NUM_THREADS:-0}"
SKIP_UPSCALE2="${SKIP_UPSCALE2:-0}"
FIELD_ORDER="${FIELD_ORDER:-auto}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chroma-shift) CHROMA_SHIFT_SX="$2"; shift 2 ;;
        --cas) CAS_STRENGTH="$2"; shift 2 ;;
        --denoise) DENOISE_STRENGTH="$2"; shift 2 ;;
        --crf) CRF="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --threads) NUM_THREADS="$2"; shift 2 ;;
        --skip-upscale2) SKIP_UPSCALE2="1"; shift ;;
        --field-order) FIELD_ORDER="$2"; shift 2 ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 input.avi [output.mkv] [--chroma-shift N] [--cas N] [--crf N]" >&2
            exit 1
            ;;
        *)
            if [[ -z "$INPUT_FILE" ]]; then
                INPUT_FILE="$1"
            elif [[ -z "$OUTPUT_FILE" ]]; then
                OUTPUT_FILE="$1"
            else
                echo "Too many positional arguments" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$INPUT_FILE" ]]; then
    echo "Usage: $0 input.avi [output.mkv] [--chroma-shift N] [--cas N] [--crf N]" >&2
    exit 1
fi

# Default output: same name with .mkv extension in same directory
if [[ -z "$OUTPUT_FILE" ]]; then
    OUTPUT_FILE="${INPUT_FILE%.*}_restored.mkv"
fi

# ============================================================
# Pre-flight checks
# ============================================================
echo "=== VHS Restoration Pipeline ==="
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo ""

# Check input exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE" >&2
    exit 1
fi

# Check VapourSynth script exists
if [[ ! -f "$VPY_SCRIPT" ]]; then
    echo "ERROR: VapourSynth script not found: $VPY_SCRIPT" >&2
    exit 1
fi

# Check required models
MISSING_MODELS=0
REQUIRED_MODELS=(
    "$MODEL_DIR/scunet_color_real_psnr.pth"
    "$MODEL_DIR/1x-BleedOut-Compact.pth"
    "$USER_MODEL_DIR/2xVHS2HD-RealPLKSR.pth"
)
if [[ "$SKIP_UPSCALE2" != "1" ]]; then
    REQUIRED_MODELS+=("$MODEL_DIR/2xLiveActionV1_SPAN.pth")
fi
for model in "${REQUIRED_MODELS[@]}"; do
    if [[ ! -f "$model" ]]; then
        echo "ERROR: Model not found: $model" >&2
        MISSING_MODELS=1
    fi
done
if [[ "$MISSING_MODELS" -eq 1 ]]; then
    echo "" >&2
    echo "Ensure all models are in $MODEL_DIR and $USER_MODEL_DIR" >&2
    echo "User model 2xVHS2HD-RealPLKSR.pth must be in $USER_MODEL_DIR" >&2
    exit 1
fi

# Check GPU
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. GPU required." >&2
    exit 1
fi

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"

# Check disk space (need ~200GB for output)
OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
AVAIL_GB=$(df -BG "$OUTPUT_DIR" 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G' || echo "0")
if [[ "$AVAIL_GB" -lt 100 ]]; then
    echo "WARNING: Only ${AVAIL_GB}GB available at $OUTPUT_DIR. Output may be 35-55GB per 3hr tape." >&2
fi

echo ""
echo "Settings:"
echo "  Chroma shift: $CHROMA_SHIFT_SX px"
echo "  CAS strength: $CAS_STRENGTH"
echo "  Denoise:      $DENOISE_STRENGTH"
echo "  CRF:          $CRF"
echo "  Preset:       $PRESET"
echo "  Threads:      $([ "$NUM_THREADS" = "0" ] && echo "auto" || echo "$NUM_THREADS")"
echo ""

# ============================================================
# Get source info for audio muxing
# ============================================================
# Get source frame rate for SAR metadata
SRC_FPS=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=r_frame_rate \
    -of csv=p=0 "$INPUT_FILE" 2>/dev/null || echo "25/1")
SRC_SAR=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=sample_aspect_ratio \
    -of csv=p=0 "$INPUT_FILE" 2>/dev/null || echo "16:15")

echo "Source: fps=$SRC_FPS, SAR=$SRC_SAR"
echo ""
echo "Starting pipeline..."
echo ""

# ============================================================
# Run pipeline: VapourSynth → FFmpeg encode
# ============================================================
export INPUT_FILE
export MODEL_DIR
export USER_MODEL_DIR
export CHROMA_SHIFT_SX
export CAS_STRENGTH
export DENOISE_STRENGTH
export NUM_THREADS
export SKIP_UPSCALE2
export FIELD_ORDER

# Select VapourSynth output method
# Prefer vspipe (R74 pip package includes it). Don't run --info as pre-check
# because it evaluates the entire script (loading all models) just to test.
VSPIPE_CMD=""
if command -v vspipe &>/dev/null; then
    VSPIPE_CMD="vspipe --y4m $VPY_SCRIPT -"
else
    echo "vspipe not found, using Python fallback..."
    # R74 API: vs.get_output(0) returns AlphaOutputTuple, .clip gets the VideoNode
    VSPIPE_CMD="python3 -c \"
import vapoursynth as vs
import sys
core = vs.core
exec(open('${VPY_SCRIPT}').read())
vs.get_output(0).clip.output(sys.stdout.buffer, y4m=True)
\""
fi

# ============================================================
# Run pipeline: VapourSynth -> FFmpeg encode
# ============================================================
# Disable set -e for the pipe so we can capture the exit code and show errors
set +e
eval "$VSPIPE_CMD" | ffmpeg -y \
    -i pipe:0 \
    -i "$INPUT_FILE" \
    -map 0:v:0 -map 1:a:0? \
    -c:v libx265 \
    -preset "$PRESET" \
    -crf "$CRF" \
    -pix_fmt yuv420p10le \
    -x265-params "sao=1:bframes=8:ref=5:aq-mode=3" \
    -aspect 4:3 \
    -sar 16:15 \
    -c:a flac \
    "$OUTPUT_FILE"

EXIT_CODE=${PIPESTATUS[0]:-1}  # Capture vspipe exit code (first in pipe)
FFMPEG_EXIT=${PIPESTATUS[1]:-1}
set -e

echo ""
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "=== VapourSynth pipeline FAILED (exit code $EXIT_CODE) ===" >&2
    exit $EXIT_CODE
elif [[ $FFMPEG_EXIT -ne 0 ]]; then
    echo "=== FFmpeg encoding FAILED (exit code $FFMPEG_EXIT) ===" >&2
    exit $FFMPEG_EXIT
else
    OUTPUT_SIZE=$(du -h "$OUTPUT_FILE" 2>/dev/null | cut -f1)
    echo "=== Pipeline complete ==="
    echo "Output: $OUTPUT_FILE ($OUTPUT_SIZE)"
    echo ""
    echo "Verify with: ffprobe -v quiet -show_format -show_streams '$OUTPUT_FILE'"
fi
