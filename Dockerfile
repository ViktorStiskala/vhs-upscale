# syntax=docker/dockerfile:1
# VHS Restoration Pipeline - Docker Image
# Base: NVIDIA CUDA with cuDNN for GPU-accelerated video processing
#
# Build:
#   docker build -t vhs-restore .
#
# One image works for ALL supported GPUs (A100, H100, RTX 5090, RTX Pro 6000).
# CUDA 12.8 supports SM120 (Blackwell) natively — no separate build needed.
#
# GPU architecture reference:
#   75=Turing, 80=Ampere (A100), 86=Ampere (RTX 3000),
#   89=Ada (RTX 4000), 90=Hopper (H100),
#   120=Blackwell desktop/pro (RTX 5090, RTX 5080, RTX Pro 6000)
#   Note: SM100 is datacenter Blackwell (B100/B200) — add only if deploying there
#
# Override CUDA version only if needed:
#   CUDA 13.0 (driver 580+): --build-arg CUDA_TAG=13.0.2-cudnn-devel-ubuntu24.04 --build-arg CU_TAG=cu130

ARG CUDA_TAG=12.8.0-cudnn-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

LABEL org.opencontainers.image.source=https://github.com/ViktorStiskala/vhs-upscale

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Broadwell-safe: target x86-64-v2 (SSE4.2 + AVX + AVX2) to avoid AVX-512
# issues on Runpod Community Cloud instances with older CPUs.
# Do NOT use -march=native — it would target the Docker build host's CPU.
ENV CFLAGS="-O2 -march=x86-64-v2 -mtune=generic"
ENV CXXFLAGS="-O2 -march=x86-64-v2 -mtune=generic"

# ============================================================
# System packages
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.12 (Ubuntu 24.04 default — matches VapourSynth R74 + PyTorch 2.11 recommended)
    python3 python3-pip python3-dev \
    # Build tools
    build-essential nasm yasm pkg-config git cmake \
    meson ninja-build autoconf automake libtool \
    # FFmpeg codec libraries
    libx264-dev libx265-dev libvpx-dev libopus-dev libvorbis-dev \
    libmp3lame-dev libfdk-aac-dev libfreetype-dev libfontconfig-dev \
    libfribidi-dev libass-dev libdav1d-dev libsvtav1enc-dev \
    libopenh264-dev libopenjp2-7-dev libwebp-dev \
    # VapourSynth plugin deps
    libfftw3-dev \
    # OpenCL + Boost (required for KNLMeansCL fallback denoiser)
    ocl-icd-opencl-dev libboost-filesystem-dev libboost-system-dev \
    # Utilities (interactive use, debugging, file transfer)
    tmux htop curl wget sudo ca-certificates gnupg locales vim jq \
    ripgrep fd-find tree unzip zip openssh-client openssh-server less man-db aria2 \
    # Source plugin deps
    libffms2-dev \
    && rm -rf /var/lib/apt/lists/*

# s5cmd (fast S3-compatible file transfer)
RUN curl -fsSL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz \
    | tar -xz -C /usr/local/bin s5cmd

# ============================================================
# Python packages - PyTorch
# ============================================================
ARG CU_TAG=cu128
# CUDA architectures for GPU-accelerated plugins (vs-nlm-cuda)
# Covers Turing through Blackwell. SM120 = RTX 5090 / RTX Pro 6000.
# SM100 = datacenter Blackwell (B100/B200) — add if deploying there.
ARG CUDA_ARCHS="75;80;86;89;90;120"

# PyTorch (no version pin — CUDA index resolves latest compatible)
RUN pip install --break-system-packages \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/${CU_TAG}

# ============================================================
# Python packages - VapourSynth ecosystem
# ============================================================

# VapourSynth R74+ (pip installable, Python 3.12+)
RUN pip install --break-system-packages vapoursynth \
    && vapoursynth config || true

# vsjetpack (QTGMC via QTempGaussMC, DFTTest, kernels, tools)
RUN pip install --break-system-packages jetpytools vsjetpack

# AI inference - vsspandrel for loading .pth models directly
# NOTE: vsspandrel is NOT published to PyPI — must install from GitHub
RUN pip install --break-system-packages git+https://github.com/TNTwise/vs-spandrel.git

# SCUNet denoiser (HolyWu's dedicated plugin)
RUN pip install --break-system-packages vsscunet

# Spandrel model loading library
RUN pip install --break-system-packages spandrel spandrel-extra-arches

# Tell VapourSynth R74 to also search the standard plugin install path.
# Meson-based plugins install to /usr/local/lib/vapoursynth/ by default,
# but R74 only auto-loads from its package subdirectory.
ENV VAPOURSYNTH_EXTRA_PLUGIN_PATH=/usr/local/lib/vapoursynth

# Also symlink ffms2 from system into VS plugin dir
RUN VS_DIR=$(python3 -c "import vapoursynth; print(vapoursynth.__path__[0])")/plugins \
    && mkdir -p "${VS_DIR}" \
    && find /usr -name "libffms2.so*" -exec ln -sf {} "${VS_DIR}/" \; 2>/dev/null || true \
    && echo "${VS_DIR}" > /tmp/vs_plugin_dir

# Install VapourSynth C headers + generate pkg-config file.
# The pip wheel doesn't ship headers, so fetch them from the matching release.
RUN VS_VERSION=$(python3 -c "import vapoursynth; print(str(vapoursynth.__version__).lstrip('R'))") \
    && VS_API=$(python3 -c "import vapoursynth; print(vapoursynth.__api_version__.api_major)") \
    && VS_LIB=$(python3 -c "import vapoursynth, os; print(os.path.dirname(vapoursynth.__path__[0]))") \
    && mkdir -p /usr/local/include/vapoursynth \
    && cd /tmp && git clone --depth 1 --branch R${VS_VERSION} https://github.com/vapoursynth/vapoursynth.git vs-headers \
    && cp vs-headers/include/*.h /usr/local/include/vapoursynth/ \
    && rm -rf /tmp/vs-headers \
    && mkdir -p /usr/local/lib/pkgconfig \
    && cat > /usr/local/lib/pkgconfig/vapoursynth.pc <<PKGEOF
prefix=/usr/local
libdir=${VS_LIB}
includedir=/usr/local/include/vapoursynth

Name: VapourSynth
Description: VapourSynth (pip)
Version: ${VS_API}
Cflags: -I\${includedir}
PKGEOF

# ============================================================
# VapourSynth native plugins (build from source)
# ============================================================

# DFTTest (required by vsdenoise/DFTTest wrapper used in QTGMC temporal denoise)
# CPU version uses FFTW3 (libfftw3-dev already installed)
RUN cd /tmp && git clone --depth 1 https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest \
    && cd VapourSynth-DFTTest \
    && meson setup build --buildtype=release \
    && ninja -C build && ninja -C build install \
    && rm -rf /tmp/VapourSynth-DFTTest

# mvtools (motion estimation for QTGMC)
# Must specify --libdir — mvtools' meson.build doesn't query VS plugindir
RUN cd /tmp && git clone --depth 1 https://github.com/dubhater/vapoursynth-mvtools \
    && cd vapoursynth-mvtools \
    && mkdir -p vapoursynth/include && ln -s /usr/local/include/vapoursynth/*.h vapoursynth/include/ \
    && meson setup build --buildtype=release --libdir=/usr/local/lib/vapoursynth \
    && ninja -C build && ninja -C build install \
    && rm -rf /tmp/vapoursynth-mvtools

# fmtconv (format conversion, chroma shift correction)
# Must specify --libdir — autotools defaults to /usr/local/lib/ which R74 doesn't search
RUN cd /tmp && git clone --depth 1 https://gitlab.com/EleonoreMizo/fmtconv.git \
    && cd fmtconv/build/unix \
    && ./autogen.sh && ./configure --libdir=/usr/local/lib/vapoursynth \
    && make -j$(nproc) && make install \
    && rm -rf /tmp/fmtconv

# znedi3 (neural net edge-directed interpolation for QTGMC)
RUN cd /tmp && git clone --recursive https://github.com/sekrit-twc/znedi3 \
    && cd znedi3 \
    && if [ "$(uname -m)" = "x86_64" ]; then make X86=1; else make; fi \
    && VS_DIR=$(cat /tmp/vs_plugin_dir) \
    && cp vsznedi3.so "${VS_DIR}/" \
    && cp nnedi3_weights.bin "${VS_DIR}/" \
    && rm -rf /tmp/znedi3

# resize2 (advanced resizer for vsjetpack)
RUN cd /tmp && git clone --depth 1 https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-resize2 \
    && cd vapoursynth-resize2 \
    && meson setup build --buildtype=release --libdir=/usr/local/lib/vapoursynth \
    && ninja -C build && ninja -C build install \
    && rm -rf /tmp/vapoursynth-resize2 \
    || echo "resize2 build failed (may be built-in to R74), skipping"

# CAS (Contrast Adaptive Sharpening — AMD FidelityFX CAS for VapourSynth)
RUN cd /tmp && git clone --depth 1 https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CAS \
    && cd VapourSynth-CAS \
    && meson setup build --buildtype=release --libdir=/usr/local/lib/vapoursynth \
    && ninja -C build && ninja -C build install \
    && rm -rf /tmp/VapourSynth-CAS \
    || echo "CAS build failed, will skip sharpening"

# vs-nlm-cuda (CUDA NLM denoiser — preferred over OpenCL KNLMeansCL)
# Must include SM90 for H100 Hopper GPU architecture
RUN cd /tmp && git clone --depth 1 https://github.com/AmusementClub/vs-nlm-cuda \
    && cd vs-nlm-cuda \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS="--use_fast_math" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    -DCMAKE_INSTALL_LIBDIR=/usr/local/lib/vapoursynth \
    && cmake --build build -j$(nproc) \
    && cmake --install build \
    && rm -rf /tmp/vs-nlm-cuda \
    || echo "vs-nlm-cuda build failed, will fall back to KNLMeansCL"

# KNLMeansCL (OpenCL fallback for chroma denoising)
# Note: uses meson build system, NOT cmake
RUN cd /tmp && git clone --depth 1 https://github.com/Khanattila/KNLMeansCL \
    && cd KNLMeansCL \
    && meson setup build --buildtype=release \
    && ninja -C build \
    && VS_DIR=$(cat /tmp/vs_plugin_dir) \
    && find build -name "*.so" -exec cp {} "${VS_DIR}/" \; \
    && rm -rf /tmp/KNLMeansCL \
    || echo "KNLMeansCL build failed, chroma denoise may be limited"

# ============================================================
# FFmpeg (built from source for architecture safety + full codec support)
# ============================================================

# NVIDIA codec headers (NVENC/NVDEC hardware acceleration)
# Use GitHub mirror — more reliable than git.videolan.org
RUN cd /tmp && git clone --depth 1 https://github.com/FFmpeg/nv-codec-headers.git \
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

# BestSource (video source plugin — built AFTER FFmpeg so it links our custom build)
# Official VapourSynth team repo, preferred over deprecated L-SMASH-Works
RUN cd /tmp && git clone --recursive --depth 1 https://github.com/vapoursynth/bestsource \
    && cd bestsource \
    && meson setup build --buildtype=release \
    && ninja -C build && ninja -C build install \
    && rm -rf /tmp/bestsource \
    || echo "bestsource build failed, will rely on ffms2"

# ============================================================
# AI models (downloaded at build time for reproducibility)
# ============================================================
RUN mkdir -p /models

# SCUNet (real-world PSNR denoiser — replaces DPIR/DRUNet)
ADD https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth \
    /models/scunet_color_real_psnr.pth

# BleedOut Compact (VHS chroma bleed/rainbow fix, 1x scale)
ADD https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/1x-BleedOut-Compact.pth \
    /models/1x-BleedOut-Compact.pth

# 2x SPAN (general-purpose second upscale pass for 4x chain)
ADD https://raw.githubusercontent.com/jcj83429/upscaling/f73a3a02874360ec6ced18f8bdd8e43b5d7bba57/2xLiveActionV1_SPAN/2xLiveActionV1_SPAN_490000.pth \
    /models/2xLiveActionV1_SPAN.pth

# ============================================================
# Pipeline scripts
# ============================================================
COPY upscale.vpy /opt/vhs-restore/upscale.vpy
COPY upscale.sh /opt/vhs-restore/upscale.sh
COPY batch_upscale.sh /opt/vhs-restore/batch_upscale.sh
COPY setup.sh /opt/vhs-restore/setup.sh
COPY start.sh /start.sh
RUN chmod +x /opt/vhs-restore/upscale.sh /opt/vhs-restore/batch_upscale.sh /opt/vhs-restore/setup.sh /start.sh

ENV PATH="/opt/vhs-restore:${PATH}"

# ============================================================
# Build-time verification
# ============================================================
RUN echo "=== Verifying installation ===" \
    && python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" \
    && python3 -c "import vapoursynth as vs; print(f'VapourSynth R{vs.__version__}')" \
    && python3 -c "from vsdeinterlace import QTempGaussMC; print('QTempGaussMC OK')" \
    && python3 -c "from vsdenoise import DFTTest; print('DFTTest OK')" \
    && python3 -c "from vsspandrel import vsspandrel; print('vsspandrel OK')" \
    && python3 -c "from vsscunet import scunet, SCUNetModel; print('vsscunet OK')" \
    && python3 -c "\
    import vapoursynth as vs; core = vs.core; \
    plugins = sorted([p for p in dir(core) if not p.startswith('_')]); \
    print('Loaded plugins:', plugins); \
    assert 'mv' in plugins, 'mvtools not loaded!'; \
    assert 'fmtc' in plugins, 'fmtconv not loaded!'; \
    assert 'znedi3' in plugins, 'znedi3 not loaded!'; \
    assert 'dfttest' in plugins, 'DFTTest native plugin not loaded!'; \
    assert 'ffms2' in plugins or 'bs' in plugins, 'No source plugin (ffms2/bestsource) loaded!'; \
    has_nlm = 'nlm_cuda' in plugins or 'knlm' in plugins; \
    print(f'NLM denoiser: {\"nlm_cuda\" if \"nlm_cuda\" in plugins else \"knlm\" if \"knlm\" in plugins else \"NONE\"}'); \
    print(f'CAS: {\"yes\" if \"cas\" in plugins else \"no\"}')" \
    && ffmpeg -version | head -1 \
    && vspipe --version || echo "WARN: vspipe not available (will use Python fallback)" \
    && ls -la /models/*.pth \
    && echo "=== All checks passed ==="

# ============================================================
# SSH server (accepts PUBLIC_KEY env var at runtime)
# ============================================================
RUN mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && ssh-keygen -A \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

EXPOSE 22

WORKDIR /workspace
CMD ["/start.sh"]
