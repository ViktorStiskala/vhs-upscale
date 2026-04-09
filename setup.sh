#!/usr/bin/env bash
# VHS Restoration Pipeline - First-run Setup Script
#
# Run this on a fresh Runpod pod to verify the installation and set up /workspace.
# This script is idempotent — safe to run multiple times.
#
# Usage: setup.sh

set -euo pipefail

# ============================================================
# ANSI colors
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
fail() { echo -e "  ${RED}✗${NC} $*"; }
warn() { echo -e "  ${YELLOW}!${NC} $*"; }
skip() { echo -e "  ${DIM}·${NC} $*"; }
header() { echo -e "\n${BOLD}${BLUE}--- $* ---${NC}"; }

echo -e "${BOLD}=== VHS Restoration Pipeline Setup ===${NC}"

# ============================================================
# 1. Verify GPU
# ============================================================
header "GPU"
if nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    echo -e "  GPUs found: ${BOLD}$GPU_COUNT${NC}"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | while read -r line; do
        echo "  $line"
    done
    if [[ $GPU_COUNT -gt 1 ]]; then
        echo ""
        echo -e "  ${GREEN}Multi-GPU detected!${NC} Use batch_upscale.sh for parallel processing:"
        echo "  batch_upscale.sh /workspace/input /workspace/output"
    fi
else
    fail "nvidia-smi not found. Is GPU available?"
    exit 1
fi

# ============================================================
# 2. Verify Python + PyTorch + Driver compatibility
# ============================================================
header "Python + PyTorch"
python3 -c "
import sys
print(f'  Python:  {sys.version.split()[0]}')
import torch
print(f'  PyTorch: {torch.__version__}')
cuda_version = torch.version.cuda or 'N/A'
print(f'  CUDA:    {cuda_version}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU in PyTorch: {torch.cuda.get_device_name(0)}')
"

# Check driver/CUDA compatibility
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 2>/dev/null || echo "")
DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || echo "")
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)

if [[ -n "$CUDA_MAJOR" && -n "$DRIVER_MAJOR" ]]; then
    # CUDA 13.0 requires R580+, CUDA 12.x requires R525+
    if [[ "$CUDA_MAJOR" -ge 13 && "$DRIVER_MAJOR" -lt 580 ]]; then
        echo ""
        fail "${RED}Driver R${DRIVER_MAJOR} does not support CUDA ${CUDA_VERSION} (requires R580+)${NC}"
        echo -e "  ${YELLOW}Use the cu128 image instead:${NC} ghcr.io/viktorstiskala/vhs-upscale:cu128"
        CUDA_COMPAT_FAIL=1
    else
        ok "Driver R${DRIVER_MAJOR} compatible with CUDA ${CUDA_VERSION}"
        CUDA_COMPAT_FAIL=0
    fi
else
    CUDA_COMPAT_FAIL=0
fi

# ============================================================
# 3. Verify VapourSynth
# ============================================================
header "VapourSynth"
python3 << 'PYEOF'
import vapoursynth as vs
print(f'  VapourSynth: R{vs.__version__}')
core = vs.core
plugins = sorted([p for p in dir(core) if not p.startswith('_')])

# Check critical plugins
critical = {
    'mv': 'mvtools (QTGMC motion estimation)',
    'fmtc': 'fmtconv (chroma shift)',
    'znedi3': 'znedi3 (QTGMC interpolation)',
    'dfttest': 'DFTTest (temporal denoise)',
    'cas': 'CAS (sharpening)',
}
missing = []
for ns, name in critical.items():
    if ns in plugins:
        print(f'  \033[0;32m✓\033[0m {name}')
    else:
        print(f'  \033[0;31m✗\033[0m {name} — MISSING')
        missing.append(name)

# Check optional plugins
optional = {
    'nlm_cuda': 'vs-nlm-cuda (CUDA chroma denoise)',
    'knlm': 'KNLMeansCL (OpenCL chroma denoise)',
    'bs': 'bestsource (video source)',
    'ffms2': 'ffms2 (video source)',
    'lsmas': 'lsmas (video source)',
}
for ns, name in optional.items():
    if ns in plugins:
        print(f'  \033[0;32m✓\033[0m {name}')
    else:
        print(f'  \033[2m·\033[0m {name} — not available (optional)')

if missing:
    print(f'\n  \033[0;31m✗ {len(missing)} critical plugins missing!\033[0m')
PYEOF

# ============================================================
# 4. Verify AI packages
# ============================================================
header "AI Packages"
python3 << 'PYEOF'
packages = [
    ('vsspandrel', 'from vsspandrel import vsspandrel', 'vsspandrel'),
    ('vsscunet', 'from vsscunet import scunet, SCUNetModel', 'vsscunet'),
    ('QTempGaussMC', 'from vsdeinterlace import QTempGaussMC', 'QTempGaussMC (QTGMC)'),
    ('DFTTest', 'from vsdenoise import DFTTest', 'DFTTest'),
]
for key, stmt, name in packages:
    try:
        exec(stmt)
        print(f'  \033[0;32m✓\033[0m {name}')
    except ImportError as e:
        print(f'  \033[0;31m✗\033[0m {name}: {e}')
PYEOF

# ============================================================
# 5. Setup /workspace directories
# ============================================================
header "Workspace"
mkdir -p /workspace/models /workspace/input /workspace/output
ok "/workspace/models/"
ok "/workspace/input/"
ok "/workspace/output/"

AVAIL=$(df -h /workspace 2>/dev/null | awk 'NR==2{print $4}' || echo "unknown")
echo -e "  Available space: ${BOLD}$AVAIL${NC}"

# ============================================================
# 6. Verify and download models
# ============================================================
header "Models"
MODEL_DIR="${MODEL_DIR:-/models}"
USER_MODEL_DIR="${USER_MODEL_DIR:-/workspace/models}"

for model in "$MODEL_DIR/scunet_color_real_psnr.pth" \
             "$MODEL_DIR/1x-BleedOut-Compact.pth" \
             "$MODEL_DIR/2xLiveActionV1_SPAN.pth"; do
    if [[ -f "$model" ]]; then
        SIZE=$(du -h "$model" | cut -f1)
        ok "$(basename "$model") ($SIZE)"
    else
        fail "$(basename "$model") — MISSING from $MODEL_DIR"
    fi
done

# User-provided model — auto-download if missing
VHS2HD_MODEL="$USER_MODEL_DIR/2xVHS2HD-RealPLKSR.pth"
VHS2HD_URL="https://huggingface.co/wavespeed/misc/resolve/main/upscalers/2xVHS2HD-RealPLKSR.pth"

if [[ -f "$VHS2HD_MODEL" ]]; then
    SIZE=$(du -h "$VHS2HD_MODEL" | cut -f1)
    ok "2xVHS2HD-RealPLKSR.pth ($SIZE)"
else
    echo -e "  ${YELLOW}Downloading${NC} 2xVHS2HD-RealPLKSR.pth..."
    if curl -fSL -o "$VHS2HD_MODEL" "$VHS2HD_URL" 2>/dev/null; then
        SIZE=$(du -h "$VHS2HD_MODEL" | cut -f1)
        ok "2xVHS2HD-RealPLKSR.pth ($SIZE) — downloaded"
    else
        fail "Failed to download 2xVHS2HD-RealPLKSR.pth"
        echo "  Place it manually in $VHS2HD_MODEL"
    fi
fi

# ============================================================
# 7. Verify FFmpeg
# ============================================================
header "FFmpeg"
ffmpeg -version 2>/dev/null | head -1 | sed 's/^/  /'
echo "  Encoders:"
ffmpeg -encoders 2>/dev/null | grep -E "libx265|libsvtav1|nvenc" | head -5 | sed 's/^/    /'

# ============================================================
# 8. Quick GPU inference test
# ============================================================
header "Quick GPU Test"
if [[ "${CUDA_COMPAT_FAIL:-0}" == "1" ]]; then
    warn "Skipped — driver/CUDA incompatible (see above)"
else
    python3 -c "
import torch
x = torch.randn(1, 3, 64, 64).cuda()
y = torch.nn.Conv2d(3, 3, 3, padding=1).cuda()(x)
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
cc = torch.cuda.get_device_capability(0)
print(f'  \033[0;32m✓\033[0m GPU inference OK ({gpu_name}, {vram_gb:.0f} GB, SM{cc[0]}{cc[1]})')
" 2>/dev/null || fail "GPU inference failed!"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo -e "${BOLD}=== Setup Complete ===${NC}"
echo ""
echo "To process a video:"
echo -e "  ${BOLD}upscale.sh /workspace/input/your_video.avi /workspace/output/restored.mkv${NC}"
echo ""
echo "Options:"
echo "  --chroma-shift -3    Chroma shift correction (PAL VHS: -2 to -6)"
echo "  --cas 0.4            CAS sharpening strength"
echo "  --crf 20             x265 quality (lower = better, default 20)"
echo "  --preset slow        x265 speed preset"
echo ""
