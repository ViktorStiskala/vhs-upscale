# Multi-GPU Batch Processing

`batch_upscale.sh` distributes multiple video files across available GPUs for parallel processing. Designed for multi-GPU pods (e.g., 2x RTX 5090).

## Usage

```bash
# Process all videos in input/ using all available GPUs
batch_upscale.sh /workspace/input /workspace/output

# With pipeline options
batch_upscale.sh /workspace/input /workspace/output --chroma-shift -3 --crf 20

# List available GPUs
batch_upscale.sh --list-gpus
```

## How It Works

1. **GPU discovery** -- detects all GPUs via `nvidia-smi`, or uses the `GPUS` env var
2. **File discovery** -- finds all video files in the input directory matching the configured extensions
3. **GPU-slot scheduling** -- maintains a pool of free GPUs. When a job finishes, its GPU is returned to the pool and the next file is assigned to it. This ensures each GPU processes exactly one file at a time, even when files have different durations
4. **Isolation** -- each job runs with `CUDA_VISIBLE_DEVICES` set to its assigned GPU
5. **Resume** -- creates `.done` markers for completed files. Re-running skips already-processed files

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPUS` | all available | Comma-separated GPU indices (e.g., `0,1`) |
| `EXTENSIONS` | `avi,mkv,mp4,mov,ts` | File extensions to process |
| `MAX_RETRIES` | `1` | Retry count per file on failure |

All `upscale.sh` options (`--chroma-shift`, `--cas`, `--crf`, etc.) are passed through.

## Logging

Per-file logs are written to `<output_dir>/.batch_logs/<filename>.log`. Check these for errors on failed files.

## Examples

```bash
# 2x RTX 5090 pod: process 10 tapes, 2 at a time
batch_upscale.sh /workspace/input /workspace/output --chroma-shift -3

# Use only GPU 0
GPUS="0" batch_upscale.sh /workspace/input /workspace/output

# Process only AVI files
EXTENSIONS="avi" batch_upscale.sh /workspace/input /workspace/output
```

With 2 GPUs and 10 tapes, 2 tapes process simultaneously. Total time is roughly 5x single-GPU time instead of 10x.
