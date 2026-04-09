# Custom Docker Image from nvidia/cuda

Build a clean image with VapourSynth R74, PyTorch, and all required plugins. This avoids most workarounds needed with the pre-built `styler00dollar/vsgan_tensorrt` image.

## Choose Base Image

Match the CUDA version to your target GPU driver. Check with `nvidia-smi` on the host:

| Driver CUDA | Base Image Tag |
|-------------|----------------|
| 12.6 | `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` |
| 12.8 | `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04` |
| 13.0 | `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` |
| 13.2 | `nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04` |

Use `devel` (not `runtime`) to get CUDA headers and nvcc for building GPU-accelerated code.

Note: the `latest` tag is deprecated for nvidia/cuda images. Always specify an explicit version.

## Dockerfile

```dockerfile
# syntax=docker/dockerfile:1
ARG CUDA_TAG=12.8.0-cudnn-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ============================================================
# System packages
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3 python3-pip python3-dev \
    # Build tools
    build-essential nasm pkg-config git \
    meson ninja-build autoconf automake libtool \
    # FFmpeg codec libraries
    libx264-dev libx265-dev libvpx-dev libopus-dev libvorbis-dev \
    libmp3lame-dev libfdk-aac-dev libfreetype-dev libfontconfig-dev \
    libfribidi-dev libass-dev libdav1d-dev libsvtav1enc-dev \
    libopenh264-dev libopenjp2-7-dev libwebp-dev \
    # VapourSynth plugin deps
    libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Python packages
# ============================================================

# Determine CUDA version for PyTorch index (e.g., 12.8 -> cu128)
ARG CUDA_VERSION=12.8
ARG CU_TAG=cu128

RUN pip install --break-system-packages \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/${CU_TAG}

# VapourSynth R74+ (pip installable)
RUN pip install --break-system-packages vapoursynth

# Configure VapourSynth (creates plugin directories, sets paths)
RUN vapoursynth config || true

# vsjetpack (QTGMC, kernels, tools) and dependencies
RUN pip install --break-system-packages jetpytools vsjetpack

# AI inference plugins
RUN pip install --break-system-packages \
    vsspandrel vsrealesrgan vsrife vsanimesr

# torch_tensorrt (optional, for TRT acceleration)
# Use --no-deps to avoid pulling conflicting torch version
RUN pip install --break-system-packages \
    torch_tensorrt --index-url https://download.pytorch.org/whl/${CU_TAG} --no-deps \
    && pip install --break-system-packages dllist packaging psutil \
    || echo "torch_tensorrt not available for ${CU_TAG}, skipping"

# ============================================================
# VapourSynth native plugins (build from source)
# ============================================================

# mvtools (motion estimation for QTGMC)
RUN cd /tmp && git clone https://github.com/dubhater/vapoursynth-mvtools \
    && cd vapoursynth-mvtools \
    && meson setup build --buildtype=release \
    && ninja -C build && ninja -C build install \
    && rm -rf /tmp/vapoursynth-mvtools

# fmtconv (format conversion)
RUN cd /tmp && git clone https://gitlab.com/EleonoreMizo/fmtconv.git \
    && cd fmtconv/build/unix \
    && ./autogen.sh && ./configure && make -j$(nproc) && make install \
    && rm -rf /tmp/fmtconv

# znedi3 (neural net interpolation for QTGMC)
RUN cd /tmp && git clone --recursive https://github.com/sekrit-twc/znedi3 \
    && cd znedi3 \
    && if [ "$(uname -m)" = "x86_64" ]; then make X86=1; else make; fi \
    && cp vsznedi3.so "$(python3 -c 'import vapoursynth; import os; print(os.path.join(os.path.dirname(vapoursynth.__file__), "plugins"))')/" \
    && cp nnedi3_weights.bin "$(python3 -c 'import vapoursynth; import os; print(os.path.join(os.path.dirname(vapoursynth.__file__), "plugins"))')/" \
    && rm -rf /tmp/znedi3

# resize2 (advanced resizer for vsjetpack)
# NOTE: R74+ may include this built-in. Check first:
#   python3 -c "import vapoursynth as vs; print(hasattr(vs.core, 'resize2'))"
# If already present, skip this block.
RUN cd /tmp && git clone https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-resize2 \
    && cd vapoursynth-resize2 \
    && meson setup build --buildtype=release \
    && ninja -C build && ninja -C build install \
    && rm -rf /tmp/vapoursynth-resize2

# ============================================================
# FFmpeg (built from source for this CPU architecture)
# ============================================================

# NVIDIA codec headers (NVENC/NVDEC)
RUN cd /tmp && git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git \
    && cd nv-codec-headers && make && make install \
    && rm -rf /tmp/nv-codec-headers

RUN cd /tmp && git clone --depth 1 https://git.ffmpeg.org/ffmpeg.git ffmpeg-src \
    && cd ffmpeg-src \
    && ./configure \
        --prefix=/usr/local \
        --enable-gpl --enable-nonfree \
        --enable-libx264 --enable-libx265 --enable-libvpx \
        --enable-libopus --enable-libvorbis --enable-libmp3lame \
        --enable-libfdk-aac --enable-libfreetype --enable-fontconfig \
        --enable-libfribidi --enable-libass \
        --enable-libdav1d --enable-libsvtav1 \
        --enable-libopenh264 --enable-libopenjpeg --enable-libwebp \
        --enable-cuda --enable-cuvid --enable-nvenc --enable-nvdec \
        --enable-optimizations --enable-pthreads --enable-runtime-cpudetect \
        --enable-shared --disable-static \
    && make -j$(nproc) \
    && make install \
    && rm -rf /tmp/ffmpeg-src \
    && ldconfig

# ============================================================
# AI models
# ============================================================
WORKDIR /workspace/tensorrt

# DPIR (DRUNet color denoiser)
ADD https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth \
    /workspace/tensorrt/drunet_color.pth

# VHS2HD model must be provided by user (COPY or mount)
# COPY 2xVHS2HD-RealPLKSR.pth /workspace/tensorrt/

# ============================================================
# Pipeline scripts
# ============================================================
COPY upscale.sh y4m_pipe.py upscale.vpy /workspace/tensorrt/
RUN chmod +x /workspace/tensorrt/upscale.sh

CMD ["bash"]
```

## Build

```bash
# For CUDA 12.8 driver (most common on current Runpod instances)
docker build \
    --build-arg CUDA_TAG=12.8.0-cudnn-devel-ubuntu24.04 \
    --build-arg CUDA_VERSION=12.8 \
    --build-arg CU_TAG=cu128 \
    -t upscale-pipeline:cu128 .

# For CUDA 13.0 driver
docker build \
    --build-arg CUDA_TAG=13.0.2-cudnn-devel-ubuntu24.04 \
    --build-arg CUDA_VERSION=13.0 \
    --build-arg CU_TAG=cu130 \
    -t upscale-pipeline:cu130 .
```

## Run

```bash
docker run --gpus all --cap-add SYS_NICE \
    -v /path/to/videos:/workspace/tensorrt/videos \
    -v /path/to/2xVHS2HD-RealPLKSR.pth:/workspace/tensorrt/2xVHS2HD-RealPLKSR.pth \
    upscale-pipeline:cu128 \
    ./upscale.sh videos/input.avi videos/output.mkv
```

## What This Eliminates

Compared to the `styler00dollar/vsgan_tensorrt` image, this custom build eliminates:

| Issue | Why it's fixed |
|-------|---------------|
| CUDA compat lib (Error 804) | Base image CUDA matches target driver |
| Illegal instruction crashes | FFmpeg built for current CPU with `--enable-runtime-cpudetect` |
| Missing resize2 plugin | VapourSynth R74 may include it; built from source as fallback |
| vspipe VSScript init failure | R74's pip install includes proper Python integration |
| PyTorch CUDA mismatch | Installed from matching cuXXX index |
| Old VapourSynth (R73) | R74 via pip with core.lazy, new packaging |
| 40GB image bloat | ~15-20GB with only what's needed |

## Adapting upscale.vpy for R74

If VapourSynth R74's `vspipe` works in the custom image (likely), you can simplify `upscale.sh` to use `vspipe` directly instead of `y4m_pipe.py`:

```bash
# In upscale.sh, replace the python3/y4m_pipe.py line with:
vspipe --y4m "$SCRIPT_DIR/upscale.vpy" - | ffmpeg ...
```

If R74 includes working `lsmas` or `bestsource` source filters, `upscale.vpy` can also be simplified to remove the FFmpeg decode pipe and use `core.lsmas.LWLibavSource()` directly.

## Runpod Template

To use this as a Runpod template:

1. Push the built image to Docker Hub or a registry
2. Create a new Runpod template with:
   - Container image: `your-registry/upscale-pipeline:cu128`
   - Docker command: leave empty (uses CMD bash)
   - Expose port: 22 (for SSH)
   - Volume mount: `/workspace`
3. Set the volume disk size to at least 50GB (for video processing scratch space)
