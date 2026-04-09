# Pipeline Architecture

## Data Flow

```
scene001.avi (720x576i, 25fps, DV PAL, BFF)
       |
       | FFmpeg subprocess (decode to raw YUV420P pipe)
       v
VapourSynth BlankClip + ModifyFrame (reads pipe frame-by-frame)
       |
       | std.SetFieldBased(1) -> marks as BFF interlaced
       v
QTempGaussMC().deinterlace(tff=False)
       |  QTGMC bob: 720x576i -> 720x576p @ 50fps
       |  CPU-only (znedi3 + mvtools + fmtconv)
       v
resize.Bicubic(format=RGBS, matrix_in_s="470bg")
       |  YUV420P8 -> RGBS (FP32), BT.601 PAL matrix
       v
vsspandrel(drunet_color.pth, trt=False)
       |  DPIR denoise: 720x576 -> 720x576, scale=1
       |  GPU, FP32 (DRUNet does not support FP16)
       v
resize.Point(format=RGBH)
       |  RGBS (FP32) -> RGBH (FP16)
       v
vsspandrel(2xVHS2HD-RealPLKSR.pth, trt=False)
       |  2x upscale: 720x576 -> 1440x1152
       |  GPU, FP16 (RealPLKSR supports FP16)
       v
resize.Bicubic(format=YUV420P10, matrix_s="470bg")
       |
       v
y4m_pipe.py (Y4M to stdout)
       |
       | pipe
       v
FFmpeg encode (libx265 CRF 18, FLAC audio from original)
       |
       v
output.mkv (1440x1152p, YUV420P10, 50fps, HEVC, SAR 16:15)
```

## Why Each Workaround Exists

### FFmpeg pipe for source (instead of lsmas/bestsource)

The `styler00dollar/vsgan_tensorrt:latest` Docker image compiles FFmpeg and VS source plugins with `-march=native` targeting AVX-512 CPUs. When the container runs on a CPU without AVX-512 (e.g., AMD EPYC 7543), these binaries crash with "Illegal instruction". Using `latest_no_avx512` tag avoids this. When the pre-built binaries crash, we decode via a separate FFmpeg process and pipe raw frames into VapourSynth through `BlankClip` + `ModifyFrame`.

### y4m_pipe.py (instead of vspipe)

The container's `vspipe` binary embeds its own Python interpreter, which has a different site-packages search path than the system Python. It cannot find the `vapoursynth` module during VSScript initialization. Rather than debugging the embedded Python's path configuration, we use a Python script that imports VapourSynth directly and writes Y4M frames to stdout.

### trt=False (instead of TensorRT)

TensorRT engine compilation requires `trt.Builder()` which needs a CUDA context. If `torch_tensorrt` was built with CUDA 13.0 but the host driver only supports CUDA 12.8, the Builder returns nullptr. PyTorch native GPU inference (`trt=False`) works regardless of this mismatch since PyTorch's CUDA runtime handles compatibility.

### CUDA compat lib disabled

NVIDIA containers include forward-compatibility libraries in `/usr/local/cuda/compat/` to make older drivers appear newer. These get registered via ldconfig. If the compat library version doesn't match the actual GPU hardware (e.g., compat lib for driver 580 on hardware running driver 570), CUDA initialization fails with Error 804. Disabling the compat ldconfig entry lets the real driver handle GPU access.

### DRUNet noise map handling

DRUNet (DPIR) expects 4-channel input: 3 RGB + 1 noise level map. Spandrel's architecture loader includes a `call_fn` that automatically generates the noise map at level 15/255 and concatenates it before calling the model. vsspandrel uses this `call_fn` in the non-TRT path. Do NOT add a manual wrapper around DRUNet - it would double-apply the noise map (5 channels, causing a crash).

### FP32 for DPIR, FP16 for VHS2HD

DRUNet's spandrel descriptor reports `supports_half=False`. Using RGBH (FP16) input would cause numerical issues. RealPLKSR supports FP16, so we convert between formats mid-pipeline to maximize GPU throughput for the upscale step.

## Performance Characteristics

- **QTGMC** is the CPU bottleneck. It uses mvtools (motion estimation) and znedi3 (neural net interpolation), both CPU-only. With 120 threads on an EPYC system, expect 5-15 fps.
- **DPIR** and **VHS2HD** run on GPU. On an RTX 5090, expect 10-30 fps per model. Since they're sequential, effective throughput is limited by the slower of the two.
- **Overall throughput** is dominated by QTGMC. GPU utilization will be intermittent as it waits for CPU-deinterlaced frames.
- **VRAM usage**: ~2-4GB for both models combined (720x576 input is small). The RTX 5090's 32GB is more than sufficient.
- **Total frames**: source frames x2 due to QTGMC bob (e.g., 278,033 source -> 556,066 output at 50fps).
