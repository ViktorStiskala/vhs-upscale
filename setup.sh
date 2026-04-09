#!/usr/bin/env bash
# VHS Restoration Pipeline - First-run Setup Script
#
# Run this on a fresh Runpod pod to verify the installation and set up /workspace.
# This script is idempotent — safe to run multiple times.
#
# Usage: setup.sh

set -euo pipefail

echo "=== VHS Restoration Pipeline Setup ==="
echo ""

# ============================================================
# 1. Verify GPU
# ============================================================
echo "--- GPU ---"
if nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    echo "GPUs found: $GPU_COUNT"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | while read line; do
        echo "  $line"
    done
    echo ""
    if [[ $GPU_COUNT -gt 1 ]]; then
        echo "Multi-GPU detected! Use batch_upscale.sh for parallel processing:"
        echo "  batch_upscale.sh /workspace/input /workspace/output"
        echo ""
    fi
else
    echo "ERROR: nvidia-smi not found. Is GPU available?" >&2
    exit 1
fi

# ============================================================
# 2. Verify Python + PyTorch
# ============================================================
echo "--- Python + PyTorch ---"
python3 -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU in PyTorch: {torch.cuda.get_device_name(0)}')
"
echo ""

# ============================================================
# 3. Verify VapourSynth
# ============================================================
echo "--- VapourSynth ---"
python3 -c "
import vapoursynth as vs
print(f'VapourSynth: R{vs.__version__}')
core = vs.core
plugins = sorted([p for p in dir(core) if not p.startswith('_')])
print(f'Loaded plugins ({len(plugins)}): {plugins}')

# Check critical plugins
critical = {
    'mv': 'mvtools (QTGMC motion estimation)',
    'fmtc': 'fmtconv (chroma shift)',
    'znedi3': 'znedi3 (QTGMC interpolation)',
    'cas': 'CAS (sharpening)',
}
missing = []
for ns, name in critical.items():
    if ns in plugins:
        print(f'  ✓ {name}')
    else:
        print(f'  ✗ {name} — MISSING')
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
        print(f'  ✓ {name}')
    else:
        print(f'  · {name} — not available (optional)')

if missing:
    print(f'\nWARNING: {len(missing)} critical plugins missing!')
"
echo ""

# ============================================================
# 4. Verify AI packages
# ============================================================
echo "--- AI Packages ---"
python3 -c "
try:
    from vsspandrel import vsspandrel
    print('✓ vsspandrel')
except ImportError as e:
    print(f'✗ vsspandrel: {e}')

try:
    from vsscunet import scunet, SCUNetModel
    print('✓ vsscunet')
except ImportError as e:
    print(f'✗ vsscunet: {e}')

try:
    from vsdeinterlace import QTempGaussMC
    print('✓ QTempGaussMC (QTGMC)')
except ImportError as e:
    print(f'✗ QTempGaussMC: {e}')

try:
    from vsdenoise import DFTTest
    print('✓ DFTTest')
except ImportError as e:
    print(f'✗ DFTTest: {e}')
"
echo ""

# ============================================================
# 5. Verify models
# ============================================================
echo "--- Models ---"
MODEL_DIR="${MODEL_DIR:-/models}"
USER_MODEL_DIR="${USER_MODEL_DIR:-/workspace/models}"

for model in "$MODEL_DIR/scunet_color_real_psnr.pth" \
             "$MODEL_DIR/1x-BleedOut-Compact.pth" \
             "$MODEL_DIR/2xLiveActionV1_SPAN.pth"; do
    if [[ -f "$model" ]]; then
        SIZE=$(du -h "$model" | cut -f1)
        echo "✓ $(basename "$model") ($SIZE)"
    else
        echo "✗ $(basename "$model") — MISSING from $MODEL_DIR"
    fi
done

# User-provided model
if [[ -f "$USER_MODEL_DIR/2xVHS2HD-RealPLKSR.pth" ]]; then
    SIZE=$(du -h "$USER_MODEL_DIR/2xVHS2HD-RealPLKSR.pth" | cut -f1)
    echo "✓ 2xVHS2HD-RealPLKSR.pth ($SIZE)"
else
    echo "✗ 2xVHS2HD-RealPLKSR.pth — MISSING"
    echo "  Place it in $USER_MODEL_DIR/2xVHS2HD-RealPLKSR.pth"
fi
echo ""

# ============================================================
# 6. Setup /workspace directories
# ============================================================
echo "--- Workspace ---"
mkdir -p /workspace/models /workspace/input /workspace/output
echo "✓ /workspace/models/"
echo "✓ /workspace/input/"
echo "✓ /workspace/output/"

# Check disk space
AVAIL=$(df -h /workspace 2>/dev/null | awk 'NR==2{print $4}' || echo "unknown")
echo "Available space: $AVAIL"
echo ""

# ============================================================
# 7. Verify FFmpeg
# ============================================================
echo "--- FFmpeg ---"
ffmpeg -version 2>/dev/null | head -1
echo "Encoders:"
ffmpeg -encoders 2>/dev/null | grep -E "libx265|libsvtav1|nvenc" | head -5
echo ""

# ============================================================
# 8. Quick GPU inference test
# ============================================================
echo "--- Quick GPU Test ---"
python3 -c "
import torch
x = torch.randn(1, 3, 64, 64).cuda()
y = torch.nn.Conv2d(3, 3, 3, padding=1).cuda()(x)
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
cc = torch.cuda.get_device_capability(0)
print(f'✓ GPU inference works ({y.shape})')
print(f'  GPU: {gpu_name}')
print(f'  VRAM: {vram_gb:.1f} GB')
print(f'  Compute capability: {cc[0]}.{cc[1]}')
arch_names = {8: 'Ampere', 9: 'Hopper/Ada', 10: 'Blackwell DC', 12: 'Blackwell'}
arch = arch_names.get(cc[0], f'Unknown (CC {cc[0]})')
print(f'  Architecture: {arch}')
if vram_gb < 16:
    print(f'  ⚠ {vram_gb:.0f}GB VRAM is tight — consider --skip-upscale2 for 2x output')
elif vram_gb < 24:
    print(f'  ⚠ {vram_gb:.0f}GB VRAM — pipeline will work but may be tight with large frames')
else:
    print(f'  ✓ {vram_gb:.0f}GB VRAM — plenty of headroom')
" 2>/dev/null && echo "" || echo "✗ GPU inference failed!"

# ============================================================
# Summary
# ============================================================
echo "=== Setup Complete ==="
echo ""
echo "To process a video:"
echo "  upscale.sh /workspace/input/your_video.avi /workspace/output/restored.mkv"
echo ""
echo "Options:"
echo "  --chroma-shift -3    Chroma shift correction (PAL VHS: -2 to -6)"
echo "  --cas 0.4            CAS sharpening strength"
echo "  --crf 20             x265 quality (lower = better, default 20)"
echo "  --preset slow        x265 speed preset"
echo ""
