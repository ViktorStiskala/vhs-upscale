# Runpod Deployment Guide

## Prerequisites

- Docker image built and pushed to a container registry (Docker Hub, GHCR, etc.)
- `2xVHS2HD-RealPLKSR.pth` model file (~29MB)
- Source AVI files on local machine or accessible URL

## 1. Build the Docker Image

```bash
cd projects/vhs-restoration

# One build works for ALL GPUs (A100, H100, RTX 5090, RTX Pro 6000)
docker build -t your-registry/vhs-restore .
docker push your-registry/vhs-restore
```

CUDA 12.8 with cu128 natively supports all target GPUs including Blackwell (SM120). The image includes compiled GPU code for SM75 through SM120 — no JIT compilation needed.

### Supported GPUs

| GPU | Architecture | SM | VRAM | Driver | Status |
|-----|-------------|-----|------|--------|--------|
| A100 SXM/PCIe | Ampere | 80 | 80 GB | R570+ | Tested |
| H100 SXM/PCIe | Hopper | 90 | 80 GB | R570+ | Tested |
| **RTX 5090** | **Blackwell** | **120** | **32 GB** | **R570+** | Supported |
| **RTX Pro 6000** | **Blackwell** | **120** | **96 GB** | **R570+** | Supported |
| RTX 4090 | Ada | 89 | 24 GB | R570+ | Supported (24GB tight) |

**Multi-GPU pods (e.g., 2x RTX 5090):** Use `batch_upscale.sh` to process multiple videos in parallel — see section 6.

**Optional: CUDA 13.0 build** (if specific 13.x features are needed):
```bash
docker build --build-arg CUDA_TAG=13.0.2-cudnn-devel-ubuntu24.04 --build-arg CU_TAG=cu130 -t vhs-restore:cu130 .
```
Requires driver R580+. Only needed for FP4/FP8 advanced CUDA features (not used in current pipeline).

## 2. Create Runpod Template

1. Go to [Runpod Templates](https://www.runpod.io/console/user/templates)
2. Click **New Template**
3. Configure:
   - **Template Name:** VHS Restoration Pipeline
   - **Container Image:** `your-registry/vhs-restore`
   - **Docker Command:** (leave empty — uses `CMD ["bash"]`)
   - **Container Disk:** 20 GB (ephemeral scratch space)
   - **Volume Disk:** 300 GB minimum (persistent `/workspace`)
   - **Volume Mount Path:** `/workspace`
   - **Expose HTTP Ports:** (none needed)
   - **Expose TCP Ports:** 22 (for SSH access)

## 3. Launch a Pod

1. Go to [Runpod GPU Cloud](https://www.runpod.io/console/gpu-cloud)
2. Select **On-Demand** (NOT Spot — spot instances terminate with 5s warning)
3. Choose GPU:
   - **H100 PCIe 80GB** ($1.99/hr) — fastest GPU inference, 16 vCPUs
   - **A100 SXM 80GB** ($1.39/hr) — best value, 16 vCPUs for QTGMC
   - **A100 PCIe 80GB** ($1.19/hr) — budget option, 8 vCPUs limit QTGMC
4. Select your template
5. Click **Deploy**

## 4. Verify Installation

SSH into the pod (or use Runpod's web terminal) and run the setup verification script:

```bash
setup.sh
```

This checks: GPU availability, PyTorch + CUDA, VapourSynth plugins, AI models, disk space. Fix any issues it reports before proceeding.

## 5. Upload Files

Use Runpod's web terminal or SSH (if enabled). For large files (40GB+ AVI), use `runpodctl` for faster transfers.

```bash
# Create directories (setup.sh already does this, but just in case)
mkdir -p /workspace/models /workspace/input /workspace/output

# Upload your VHS2HD model (~29MB)
# Option A: runpodctl (fastest, Runpod's own tool — install on local machine first)
runpodctl send 2xVHS2HD-RealPLKSR.pth  # On local machine — gives you a code
runpodctl receive <code>                 # On pod — downloads the file
mv 2xVHS2HD-RealPLKSR.pth /workspace/models/

# Option B: Download from URL (if hosted somewhere)
wget -O /workspace/models/2xVHS2HD-RealPLKSR.pth "<your-model-url>"

# Option C: rclone from cloud storage (S3, GCS, etc.)
# rclone copy remote:bucket/2xVHS2HD-RealPLKSR.pth /workspace/models/

# Upload source video (40GB+ — runpodctl recommended)
runpodctl send scene0001.avi             # On local machine
runpodctl receive <code>                  # On pod
mv scene0001.avi /workspace/input/
```

**Transfer time estimates:**
- 29MB model: seconds via any method
- 40GB AVI via `runpodctl`: ~10-20 minutes (Runpod network)
- 40GB AVI via SCP at 20 Mbps home upload: ~4-5 hours

## 6. Run the Pipeline

```bash
# Basic usage
upscale.sh /workspace/input/scene0001.avi /workspace/output/scene0001_restored.mkv

# With chroma shift correction (typical PAL VHS: -3)
upscale.sh /workspace/input/scene0001.avi /workspace/output/scene0001_restored.mkv \
    --chroma-shift -3

# Full options
upscale.sh /workspace/input/scene0001.avi /workspace/output/scene0001_restored.mkv \
    --chroma-shift -3 \
    --cas 0.4 \
    --denoise 3.0 \
    --crf 20 \
    --preset slow
```

### Available options

| Flag | Default | Description |
|------|---------|-------------|
| `--chroma-shift` | 0 | Chroma shift correction in pixels (-2 to -6 for PAL VHS) |
| `--cas` | 0.4 | CAS sharpening strength (0.0 = off, 1.0 = max) |
| `--denoise` | 3.0 | NLM chroma denoise strength |
| `--crf` | 20 | x265 CRF (lower = higher quality, bigger file) |
| `--preset` | slow | x265 preset (slower = better compression) |
| `--threads` | auto | VapourSynth thread count |
| `--skip-upscale2` | off | Skip second upscale pass (output 1440x1152 instead of 2880x2304) |
| `--field-order` | auto | Field order: auto, bff, tff, progressive (auto-detects from source) |

### Multi-GPU batch processing (2x RTX 5090 or multi-GPU pods)

For pods with multiple GPUs, use `batch_upscale.sh` to process videos in parallel:

```bash
# Process all videos in input/ using all available GPUs
batch_upscale.sh /workspace/input /workspace/output --chroma-shift -3

# List available GPUs
batch_upscale.sh --list-gpus

# Use specific GPUs only
GPUS="0,1" batch_upscale.sh /workspace/input /workspace/output

# With 2x RTX 5090: each GPU processes one video simultaneously
# Completed files are marked with .done — re-running skips them
```

**How it works:**
- Discovers all GPUs via `nvidia-smi`
- Assigns each video to a GPU round-robin
- Runs one `upscale.sh` per GPU in parallel via `CUDA_VISIBLE_DEVICES`
- Logs per-file in `output/.batch_logs/`
- Creates `.done` markers — re-running safely skips completed files
- Supports all `upscale.sh` flags (passed through)

**2x RTX 5090 example:** With 2 GPUs and 10 tapes, 2 tapes process simultaneously. Total time ≈ 5 × single-GPU time instead of 10×.

## 7. Expected Output

- **Input:** 720x576i PAL DV, 25fps interlaced, ~40GB AVI
- **Output:** 2880x2304p, 50fps progressive, YUV420P10, HEVC in MKV
- **Size:** ~35-55GB at CRF 20 (default), larger at lower CRF values
- **Aspect ratio:** 4:3 (SAR 16:15 preserved)

## 8. Cost Estimates

Processing ~3 hours of PAL VHS content (278K source frames -> 556K output frames after bob):

The pipeline throughput is limited by the **slowest stage**. With 4x upscaling (SPAN at 1440x1152), the GPU upscale step at ~5-10fps is likely the bottleneck.

| Scenario | GPU | Throughput | Wall Time | Est. Cost |
|----------|-----|-----------|-----------|-----------|
| **4x upscale** | H100 PCIe | ~5-10 fps | 15-31 hours | $30-62 |
| **4x upscale** | RTX 5090 | ~8-15 fps | 10-20 hours | varies |
| **4x upscale** | 2x RTX 5090 | ~8-15 fps/GPU | 10-20 hrs (2 videos) | varies |
| **2x only** | H100 PCIe | ~10-15 fps | 10-15 hours | $20-30 |
| **Future: TRT** | any | ~15-30 fps | 5-10 hours | $10-20 |

| GPU | Rate | Est. Cost (3hr source, 4x) |
|-----|------|---------------------------|
| A100 SXM 80GB | $1.39/hr | $21-43 |
| H100 PCIe 80GB | $1.99/hr | $30-62 |

**Tips to reduce cost:**
- Use `SKIP_UPSCALE2=1` for 2x output (1440x1152) — roughly halves processing time
- Use A100 SXM instead of H100 — 30% cheaper, QTGMC CPU performance is similar
- Consider enabling TensorRT (`trt=True`) in a future iteration for 2-4x GPU speedup

**Output file sizes (CRF 20):**
- ~35-55GB per 3hr tape (HEVC 2880x2304 50fps)
- ~20-30GB per 3hr tape with 2x only (HEVC 1440x1152 50fps)

## 9. Troubleshooting

### "Model not found" error
Ensure `2xVHS2HD-RealPLKSR.pth` is in `/workspace/models/`. Built-in models are in `/models/`.

### Out of disk space
Increase the network volume size. Output files can be 100-200GB. Use at least 300GB total.

### GPU out of memory
Pipeline peaks at ~4-6GB VRAM — well within 32GB (RTX 5090) or 80GB (H100/A100). If OOM occurs, reduce VapourSynth prefetch:
```bash
# In upscale.vpy, add after core = vs.core:
core.max_cache_size = 32000  # Limit to 32GB
```

### QTGMC very slow
QTGMC is CPU-bound. More vCPUs = faster. A100 PCIe (8 vCPUs) will be ~2x slower than SXM (16 vCPUs).

### Pod terminated unexpectedly
Always use **On-Demand** pods. Spot instances can be terminated with 5s warning, killing your encode.

### vspipe not working
If vspipe fails, the script automatically falls back to a Python Y4M pipe. No action needed.

## 10. After Processing

Download the output:
```bash
# From pod to local machine
scp -P <port> root@<pod-ip>:/workspace/output/scene0001_restored.mkv .
```

**Stop the pod** when done to avoid charges. Your `/workspace` data persists.

To process another file, just start the pod again — the Docker image and models are already there.
