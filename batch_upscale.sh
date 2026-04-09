#!/usr/bin/env bash
# VHS Restoration Pipeline - Multi-GPU Batch Processing
#
# Distributes multiple video files across available GPUs for parallel processing.
# Designed for workstations with 2x RTX 5090 or similar multi-GPU setups.
#
# Usage:
#   ./batch_upscale.sh /path/to/input/dir [/path/to/output/dir] [options]
#   ./batch_upscale.sh --list-gpus
#
# Examples:
#   # Process all .avi files in input/ using all available GPUs
#   ./batch_upscale.sh /workspace/input /workspace/output
#
#   # Process with specific options
#   ./batch_upscale.sh /workspace/input /workspace/output --chroma-shift -3 --crf 20
#
#   # Use only GPUs 0 and 1
#   GPUS="0,1" ./batch_upscale.sh /workspace/input /workspace/output
#
# Environment variables:
#   GPUS             - Comma-separated GPU indices (default: all available)
#   EXTENSIONS       - File extensions to process (default: "avi,mkv,mp4,mov,ts")
#   MAX_RETRIES      - Retry count per file on failure (default: 1)
#
# All upscale.sh options (--chroma-shift, --cas, --crf, etc.) are passed through.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSCALE_SCRIPT="${SCRIPT_DIR}/upscale.sh"

# ============================================================
# Parse arguments
# ============================================================
INPUT_DIR=""
OUTPUT_DIR=""
PASSTHROUGH_ARGS=()
GPUS="${GPUS:-}"
EXTENSIONS="${EXTENSIONS:-avi,mkv,mp4,mov,ts}"
MAX_RETRIES="${MAX_RETRIES:-1}"

# Handle --list-gpus
if [[ "${1:-}" == "--list-gpus" ]]; then
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    exit 0
fi

# Parse positional and passthrough args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --extensions) EXTENSIONS="$2"; shift 2 ;;
        --retries) MAX_RETRIES="$2"; shift 2 ;;
        --chroma-shift|--cas|--denoise|--crf|--preset|--threads|--field-order)
            PASSTHROUGH_ARGS+=("$1" "$2"); shift 2 ;;
        --skip-upscale2)
            PASSTHROUGH_ARGS+=("$1"); shift ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [[ -z "$INPUT_DIR" ]]; then
                INPUT_DIR="$1"
            elif [[ -z "$OUTPUT_DIR" ]]; then
                OUTPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$INPUT_DIR" ]]; then
    echo "Usage: $0 /path/to/input/dir [/path/to/output/dir] [options]" >&2
    echo "       $0 --list-gpus" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}_restored}"
mkdir -p "$OUTPUT_DIR"

# ============================================================
# Discover GPUs
# ============================================================
if [[ -z "$GPUS" ]]; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    GPU_LIST=($(nvidia-smi --query-gpu=index --format=csv,noheader))
else
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
    GPU_COUNT=${#GPU_LIST[@]}
fi

echo "=== VHS Restoration - Multi-GPU Batch Processing ==="
echo ""
echo "GPUs:       ${GPU_COUNT} (indices: ${GPU_LIST[*]})"
echo "Input:      $INPUT_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Extensions: $EXTENSIONS"
echo "Retries:    $MAX_RETRIES"
if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    echo "Options:    ${PASSTHROUGH_ARGS[*]}"
fi
echo ""

# ============================================================
# Discover input files
# ============================================================
IFS=',' read -ra EXT_LIST <<< "$EXTENSIONS"
INPUT_FILES=()
for ext in "${EXT_LIST[@]}"; do
    while IFS= read -r -d '' file; do
        INPUT_FILES+=("$file")
    done < <(find "$INPUT_DIR" -maxdepth 1 -type f -iname "*.${ext}" -print0 | sort -z)
done

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
    echo "No video files found in $INPUT_DIR with extensions: $EXTENSIONS" >&2
    exit 1
fi

echo "Found ${#INPUT_FILES[@]} files to process:"
for f in "${INPUT_FILES[@]}"; do
    echo "  $(basename "$f")"
done
echo ""

# ============================================================
# GPU assignment and parallel processing
# ============================================================
LOG_DIR="${OUTPUT_DIR}/.batch_logs"
mkdir -p "$LOG_DIR"

# Track results
declare -A JOB_PIDS
declare -A JOB_FILES
declare -A JOB_GPUS
COMPLETED=0
FAILED=0
SKIPPED=0

process_file() {
    local input_file="$1"
    local gpu_id="$2"
    local basename_no_ext="${input_file%.*}"
    basename_no_ext="$(basename "$basename_no_ext")"
    local output_file="${OUTPUT_DIR}/${basename_no_ext}_restored.mkv"
    local log_file="${LOG_DIR}/${basename_no_ext}.log"
    local done_marker="${OUTPUT_DIR}/.${basename_no_ext}.done"

    # Skip if already processed
    if [[ -f "$done_marker" ]]; then
        echo "[GPU $gpu_id] SKIP: $(basename "$input_file") (already processed)"
        return 2  # skip code
    fi

    echo "[GPU $gpu_id] START: $(basename "$input_file") -> $(basename "$output_file")"
    echo "[GPU $gpu_id] Log: $log_file"

    # Run upscale.sh on the assigned GPU
    CUDA_VISIBLE_DEVICES="$gpu_id" \
        "$UPSCALE_SCRIPT" "$input_file" "$output_file" \
        "${PASSTHROUGH_ARGS[@]}" \
        > "$log_file" 2>&1

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        # Create done marker
        echo "completed=$(date -Iseconds)" > "$done_marker"
        echo "[GPU $gpu_id] DONE: $(basename "$input_file")"
        return 0
    else
        echo "[GPU $gpu_id] FAILED: $(basename "$input_file") (exit $exit_code, see $log_file)" >&2
        return 1
    fi
}

# GPU-slot scheduling: assign work to whichever GPU finishes first
FREE_GPUS=("${GPU_LIST[@]}")

collect_finished() {
    # Collect a finished job and return its GPU to the free pool.
    # Uses wait -n -p (bash 5.1+) with fallback for older bash.
    local FINISHED_PID="" local_exit=0

    if wait -n -p FINISHED_PID 2>/dev/null; then
        local_exit=0
    else
        local_exit=$?
    fi

    if [[ -n "$FINISHED_PID" ]] && [[ -n "${JOB_PIDS[$FINISHED_PID]+x}" ]]; then
        if [[ $local_exit -eq 0 ]]; then ((COMPLETED++)); elif [[ $local_exit -eq 2 ]]; then ((SKIPPED++)); else ((FAILED++)); fi
        FREE_GPUS+=("${JOB_GPUS[$FINISHED_PID]}")
        unset "JOB_PIDS[$FINISHED_PID]"
        unset "JOB_GPUS[$FINISHED_PID]"
    else
        # Fallback: scan all PIDs (older bash or wait -p unavailable)
        for pid in "${!JOB_PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null; local_exit=$?
                if [[ $local_exit -eq 0 ]]; then ((COMPLETED++)); elif [[ $local_exit -eq 2 ]]; then ((SKIPPED++)); else ((FAILED++)); fi
                FREE_GPUS+=("${JOB_GPUS[$pid]}")
                unset "JOB_PIDS[$pid]"
                unset "JOB_GPUS[$pid]"
            fi
        done
    fi
}

for input_file in "${INPUT_FILES[@]}"; do
    # Wait until a GPU is free
    while [[ ${#FREE_GPUS[@]} -eq 0 ]]; do
        collect_finished
    done

    # Pop the first free GPU
    current_gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")

    # Launch in background
    process_file "$input_file" "$current_gpu" &
    local_pid=$!
    JOB_PIDS[$local_pid]=1
    JOB_FILES[$local_pid]="$input_file"
    JOB_GPUS[$local_pid]="$current_gpu"
done

# Wait for all remaining jobs
for pid in "${!JOB_PIDS[@]}"; do
    wait "$pid" 2>/dev/null
    local_exit=$?
    if [[ $local_exit -eq 0 ]]; then
        ((COMPLETED++))
    elif [[ $local_exit -eq 2 ]]; then
        ((SKIPPED++))
    else
        ((FAILED++))
    fi
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "=== Batch Processing Complete ==="
echo "Total:     ${#INPUT_FILES[@]} files"
echo "Completed: $COMPLETED"
echo "Skipped:   $SKIPPED (already processed)"
echo "Failed:    $FAILED"
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo "Failed files (check logs in $LOG_DIR):"
    for log in "$LOG_DIR"/*.log; do
        if [[ -f "$log" ]]; then
            basename_log="$(basename "$log" .log)"
            if [[ ! -f "${OUTPUT_DIR}/.${basename_log}.done" ]]; then
                echo "  $basename_log — $log"
            fi
        fi
    done
    echo ""
    echo "Re-run this command to retry failed files (completed files will be skipped)."
    exit 1
fi

echo "All files processed successfully."
echo "Output: $OUTPUT_DIR"
