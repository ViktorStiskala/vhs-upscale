# Environment Setup Guide

Complete setup from a fresh `styler00dollar/vsgan_tensorrt` container. Each step includes detection, fix, and verification.

## Prerequisites

- NVIDIA GPU (any architecture: Ampere, Ada Lovelace, Hopper, Blackwell)
- Docker with NVIDIA Container Toolkit
- Recommended: `docker run --cap-add SYS_NICE` for x265 NUMA optimization

## Step 1: Choose Docker Image

```bash
# Check if your CPU supports AVX-512
avx512_count=$(grep -c avx512 /proc/cpuinfo 2>/dev/null || echo 0)
if [ "$avx512_count" -gt 0 ]; then
    echo "Use: styler00dollar/vsgan_tensorrt:latest"
else
    echo "Use: styler00dollar/vsgan_tensorrt:latest_no_avx512"
fi
```

The `_no_avx512` tag builds FFmpeg and VS plugins without AVX-512 instructions, avoiding "Illegal instruction" crashes on CPUs like AMD EPYC 7xx3, consumer Ryzen/Intel, etc.

## Step 2: Fix CUDA Compatibility Library

The container may bundle a CUDA forward-compatibility library that overrides the host's real GPU driver. This causes "Error 804: forward compatibility was attempted on non supported HW".

```bash
# Detect: check if compat lib shadows real driver
compat_lib=$(ldconfig -p 2>/dev/null | grep "libcuda.so.1" | head -1)
if echo "$compat_lib" | grep -q "compat"; then
    echo "WARNING: CUDA compat lib is overriding real driver"

    # Fix: disable the compat ldconfig entry
    for f in /etc/ld.so.conf.d/*compat*; do
        mv "$f" "${f}.disabled" 2>/dev/null
    done
    ldconfig

    echo "Fixed. Verify:"
fi

# Verify: should show real driver path (e.g., /lib/x86_64-linux-gnu/libcuda.so.1)
ldconfig -p | grep "libcuda.so.1"
nvidia-smi | head -5
```

## Step 3: Bootstrap pip

```bash
# Detect
if ! command -v pip &>/dev/null; then
    curl -sL https://bootstrap.pypa.io/get-pip.py | python3 - --break-system-packages
fi
pip --version
```

## Step 4: Install PyTorch (matching driver CUDA version)

```bash
# Detect driver CUDA version
cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "Driver CUDA: $cuda_version"

# Map to PyTorch index (e.g., 12.8 -> cu128, 13.0 -> cu130)
cuda_major=$(echo $cuda_version | cut -d. -f1)
cuda_minor=$(echo $cuda_version | cut -d. -f2)
cu_tag="cu${cuda_major}${cuda_minor}"
echo "PyTorch index: $cu_tag"

# Install
pip install --break-system-packages \
    torch torchvision \
    --index-url "https://download.pytorch.org/whl/${cu_tag}"

# Clean old torch from site-packages if it shadows dist-packages
if [ -d /usr/local/lib/python3.12/site-packages/torch ]; then
    rm -rf /usr/local/lib/python3.12/site-packages/torch \
           /usr/local/lib/python3.12/site-packages/torch-*.dist-info \
           /usr/local/lib/python3.12/site-packages/torchvision \
           /usr/local/lib/python3.12/site-packages/torchvision-*.dist-info
fi

# Clean old NVIDIA cu12/cu13 packages from site-packages
rm -rf /usr/local/lib/python3.12/site-packages/nvidia \
       /usr/local/lib/python3.12/site-packages/nvidia_*.dist-info

# Verify
PYTHONPATH=/usr/local/lib/python3.12/site-packages python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))
"
```

## Step 5: Install torch_tensorrt (optional)

TensorRT acceleration. May not work if torch_tensorrt's CUDA version exceeds the driver's.

```bash
# Install matching version (use --no-deps to avoid pulling wrong torch)
pip install --break-system-packages \
    torch_tensorrt --index-url "https://download.pytorch.org/whl/${cu_tag}" --no-deps

# Install required deps
pip install --break-system-packages dllist packaging psutil

# Verify (warnings about TRTLLM are normal)
PYTHONPATH=/usr/local/lib/python3.12/site-packages python3 -c "
import torch_tensorrt; print('Torch-TRT:', torch_tensorrt.__version__)
" 2>&1 | grep -v "TRTLLM\|WARNING\|Unable to import"
```

If torch_tensorrt for your `cu_tag` doesn't exist, use `--no-deps` with a nearby version (e.g., cu130 with cu128 driver works for inference, but TRT Builder will fail). In that case, use `trt=False` in vsspandrel calls.

## Step 6: Install vsjetpack (QTGMC + dependencies)

```bash
pip install --break-system-packages jetpytools
pip install --break-system-packages --no-deps vsjetpack
# --no-deps prevents pip from installing PyPI's VapourSynth package
# which would overwrite the container's pre-built VapourSynth

# Verify
PYTHONPATH=/usr/local/lib/python3.12/site-packages python3 -c "
from vsdeinterlace import QTempGaussMC; print('QTGMC: OK')
"
```

## Step 7: Fix VapourSynth Plugin Symlinks

mvtools and fmtconv may be installed but not in the VS autoload directory.

```bash
# Detect and fix
vs_plugins=/usr/local/lib/vapoursynth

for lib in libmvtools.so libfmtconv.so; do
    if [ ! -f "$vs_plugins/$lib" ]; then
        src=$(find /usr/local/lib -name "$lib" -not -path "*/vapoursynth/*" 2>/dev/null | head -1)
        if [ -n "$src" ]; then
            ln -sf "$src" "$vs_plugins/$lib"
            echo "Linked: $lib"
        fi
    fi
done

# Verify
PYTHONPATH=/usr/local/lib/python3.12/site-packages python3 -c "
import vapoursynth as vs; core = vs.core
print('mv:', hasattr(core, 'mv'), '| fmtc:', hasattr(core, 'fmtc'))
"
```

## Step 8: Build znedi3 (NNEDI3 for QTGMC)

```bash
cd /tmp && git clone --recursive https://github.com/sekrit-twc/znedi3
cd /tmp/znedi3

# Build (X86=1 for x86_64, omit for other architectures)
if uname -m | grep -q x86_64; then
    make X86=1
else
    make
fi

cp vsznedi3.so /usr/local/lib/vapoursynth/
cp nnedi3_weights.bin /usr/local/lib/vapoursynth/
# IMPORTANT: weights file MUST be in same directory as .so

# Verify
PYTHONPATH=/usr/local/lib/python3.12/site-packages python3 -c "
import vapoursynth as vs; print('znedi3:', hasattr(vs.core, 'znedi3'))
"
```

## Step 9: Build vapoursynth-resize2

Required by vsjetpack's vskernels. VapourSynth R74+ includes this built-in, so skip if on R74+.

```bash
# Check if already available
PYTHONPATH=/usr/local/lib/python3.12/site-packages python3 -c "
import vapoursynth as vs; print('resize2:', hasattr(vs.core, 'resize2'))
" 2>&1 | grep -q "True" && echo "resize2 already available" && exit 0

# Install build tools
apt-get update -qq && apt-get install -y --no-install-recommends meson ninja-build

# Install VS headers if missing
if [ ! -f /usr/local/include/vapoursynth/VapourSynth4.h ]; then
    mkdir -p /usr/local/include/vapoursynth
    # Copy from znedi3's bundled headers
    cp /tmp/znedi3/vsxx/vapoursynth/*.h /usr/local/include/vapoursynth/
fi

# Create vapoursynth.pc if missing
if ! pkg-config --exists vapoursynth 2>/dev/null; then
    vs_version=$(PYTHONPATH=/usr/local/lib/python3.12/site-packages python3 -c "
import vapoursynth as vs; print(vs.core.core_version.release_major)" 2>/dev/null || echo 73)
    cat > /usr/local/lib/pkgconfig/vapoursynth.pc << EOF
prefix=/usr/local
includedir=\${prefix}/include/vapoursynth
Name: vapoursynth
Description: VapourSynth API
Version: $vs_version
Cflags: -I\${includedir}
EOF
fi

# Build
cd /tmp && git clone https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-resize2
cd /tmp/vapoursynth-resize2
meson setup build --buildtype=release
ninja -C build
cp build/libresize2.so /usr/local/lib/vapoursynth/

# Verify
PYTHONPATH=/usr/local/lib/python3.12/site-packages python3 -c "
import vapoursynth as vs; print('resize2:', hasattr(vs.core, 'resize2'))
"
```

## Step 10: Build FFmpeg from Source (if needed)

Only needed if the pre-installed FFmpeg crashes with "Illegal instruction". Test first:

```bash
ffmpeg -f lavfi -i nullsrc=s=1x1:d=0.04 -frames:v 1 -f null - 2>&1
echo "Exit: $?"  # 0 = OK, 132 = Illegal instruction
```

If it crashes, build from source:

```bash
# Install build dependencies
apt-get install -y --no-install-recommends \
    nasm pkg-config \
    libx264-dev libx265-dev libvpx-dev libopus-dev libvorbis-dev \
    libmp3lame-dev libfdk-aac-dev libfreetype-dev libfontconfig-dev \
    libfribidi-dev libass-dev libdav1d-dev libsvtav1enc-dev \
    libopenh264-dev libopenjp2-7-dev libwebp-dev

# Fix dpkg if interrupted
dpkg --configure -a 2>/dev/null

# Install NVIDIA codec headers
cd /tmp && git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd /tmp/nv-codec-headers && make && make install

# Find CUDA headers (may be in pip-installed nvidia packages)
cuda_inc=$(find /usr/local/lib/python3.12/dist-packages/nvidia -name "cuda.h" -printf "%h" -quit 2>/dev/null)
cuda_inc=${cuda_inc:-/usr/local/cuda/include}

# Clone and build FFmpeg
cd /tmp && git clone --depth 1 https://git.ffmpeg.org/ffmpeg.git ffmpeg-src
cd /tmp/ffmpeg-src

./configure \
    --prefix=/usr/local \
    --enable-gpl --enable-nonfree \
    --enable-libx264 --enable-libx265 --enable-libvpx \
    --enable-libopus --enable-libvorbis --enable-libmp3lame \
    --enable-libfdk-aac --enable-libfreetype --enable-fontconfig \
    --enable-libfribidi --enable-libass \
    --enable-libdav1d --enable-libsvtav1 \
    --enable-libopenh264 --enable-libopenjpeg --enable-libwebp \
    --enable-cuda --enable-cuvid --enable-nvenc --enable-nvdec \
    --extra-cflags="-I${cuda_inc} -I/usr/local/include" \
    --extra-ldflags="-L/usr/local/cuda/targets/x86_64-linux/lib" \
    --enable-optimizations --enable-pthreads --enable-runtime-cpudetect \
    --enable-shared --disable-static

make -j$(nproc)
make install
ldconfig

# Verify
ffmpeg -version | head -2
```

## Step 11: Download Models

```bash
cd /workspace/tensorrt

# DPIR (DRUNet color denoiser)
wget -O drunet_color.pth \
    https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth

# VHS2HD (2x upscaler) - place your model file here
# ls -lh 2xVHS2HD-RealPLKSR.pth
```

## Step 12: Verify Everything

```bash
PYTHONPATH=/usr/local/lib/python3.12/site-packages python3 -c "
import vapoursynth as vs
core = vs.core

# VS plugins
for p in ['mv', 'fmtc', 'znedi3', 'resize2', 'std', 'resize']:
    print(f'{p}: {hasattr(core, p)}')

# Python packages
from vsdeinterlace import QTempGaussMC; print('QTGMC: OK')
from vsspandrel import vsspandrel; print('vsspandrel: OK')

# PyTorch + GPU
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}')
print(f'Arch: {torch.cuda.get_arch_list()}')

# TensorRT
import tensorrt; print(f'TensorRT: {tensorrt.__version__}')

# Models
import os
for m in ['drunet_color.pth', '2xVHS2HD-RealPLKSR.pth']:
    path = os.path.join('/workspace/tensorrt', m)
    exists = os.path.isfile(path)
    size = os.path.getsize(path) // (1024*1024) if exists else 0
    print(f'{m}: {\"OK\" if exists else \"MISSING\"} ({size}MB)')

print('\\nAll checks passed!')
"

# Test FFmpeg
ffmpeg -f lavfi -i nullsrc=s=1x1:d=0.04 -frames:v 1 -f null - 2>/dev/null && echo "FFmpeg: OK"
```
