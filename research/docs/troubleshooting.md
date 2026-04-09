# Troubleshooting

## Error 804: forward compatibility was attempted on non supported HW

**Cause**: CUDA forward-compatibility library in `/usr/local/cuda/compat/` overrides the real GPU driver via ldconfig.

**Diagnose**:
```bash
ldconfig -p | grep "libcuda.so.1"
# If the first result points to a "compat" path, that's the problem
```

**Fix**:
```bash
for f in /etc/ld.so.conf.d/*compat*; do mv "$f" "${f}.disabled"; done
ldconfig
```

**Verify**: `nvidia-smi` shows correct CUDA version; `ldconfig -p | grep libcuda.so.1` points to `/lib/x86_64-linux-gnu/`.

---

## Illegal instruction (core dumped)

**Cause**: Binaries compiled with AVX-512 running on a CPU without AVX-512.

**Diagnose**:
```bash
grep -c avx512 /proc/cpuinfo  # 0 = no AVX-512
```

**Fix** (choose one):
- Use `styler00dollar/vsgan_tensorrt:latest_no_avx512` Docker tag
- Rebuild FFmpeg from source (see `setup.md` Step 10)
- The `upscale.vpy` script already works around source filter crashes by using FFmpeg pipe

---

## Failed to initialize VSScript

**Cause**: `vspipe`'s embedded Python cannot find the `vapoursynth` module. The embedded Python searches `/usr/lib/python3.12/` but VS is in `/usr/local/lib/python3.12/site-packages/`.

**Fix**: Use `y4m_pipe.py` instead of `vspipe`. The bash script `upscale.sh` already does this.

---

## No attribute with the name resize2 exists

**Cause**: VapourSynth R73 doesn't include the `resize2` plugin. vsjetpack requires it.

**Fix**: Build and install `vapoursynth-resize2` from source (see `setup.md` Step 9).

**Note**: VapourSynth R74+ includes `resize2` built-in. If the Docker image upgrades to R74, this step becomes unnecessary.

---

## RuntimeError: expected input to have 4 channels, but got 3

**Cause**: DRUNet model expects 4 channels (RGB + noise map) but received raw 3-channel RGB. This happens when using `trt=True` because vsspandrel's TRT compilation path bypasses spandrel's `call_fn` that auto-generates the noise map.

**Fix**: Use `trt=False` for DRUNet/DPIR. Spandrel's `call_fn` handles the 4th channel automatically in PyTorch mode.

---

## RuntimeError: expected input to have 4 channels, but got 5

**Cause**: A DRUNetWrapper was added on top of spandrel's built-in noise map generation, doubling the noise channel.

**Fix**: Remove any DRUNetWrapper. Use `trt=False` and let spandrel handle it.

---

## TypeError: pybind11::init(): factory function returned nullptr (TRT Builder)

**Cause**: `torch_tensorrt`'s CUDA version exceeds the host driver's CUDA capability. e.g., torch_tensorrt built with CUDA 13.0 on a driver that only supports CUDA 12.8.

**Fix**: Use `trt=False` in all `vsspandrel()` calls. PyTorch native GPU inference works regardless.

---

## The NVIDIA driver on your system is too old (found version XXXXX)

**Cause**: PyTorch's CUDA runtime version is newer than what the driver supports. e.g., torch cu130 on a driver supporting CUDA 12.8.

**Fix**: Install PyTorch matching your driver's CUDA version:
```bash
cuda_ver=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
# e.g., 12.8 -> cu128
pip install --break-system-packages torch torchvision \
    --index-url https://download.pytorch.org/whl/cu${cuda_ver//./}
```

---

## set_mempolicy: Operation not permitted

**Cause**: x265 encoder tries to use NUMA memory policies, blocked by Docker's seccomp profile.

**Fix**: Start container with `--cap-add SYS_NICE` or `--security-opt seccomp=unconfined`.

**Impact**: Cosmetic only. Encoding works correctly without NUMA optimization.

---

## AttributeError: 'VideoOutputTuple' object has no attribute 'width'

**Cause**: `vs.get_output(0)` returns a tuple in newer VapourSynth, not a clip directly.

**Fix**: Unwrap: `clip = vs.get_output(0)[0]`

---

## GPU shows 0% utilization during processing

**Cause**: QTGMC (CPU deinterlacing) is the pipeline bottleneck. GPU is idle while waiting for CPU frames.

**Diagnosis**: `watch -n1 nvidia-smi` will show GPU spikes when DPIR/VHS2HD process frames, but most time is CPU-bound QTGMC.

**Optimization**: No fix for QTGMC's CPU dependency. Ensure the container has sufficient CPU cores allocated (32+ recommended).
