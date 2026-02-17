#!/bin/sh
# Launch V-JEPA2 embedding extraction in background
# Run from repository root: sh scripts/run_extraction.sh

# Exit on error (before backgrounding)
set -e

# Change to repository root directory
cd /home/haohw/vjepa2 || exit 1

# Default settings
MODEL_SIZE="giant"   # Options: large (300M), giant (1B)
POOLING="mean"       # Options: mean, max
BATCH_SIZE=16         # 4 pairs = 8 videos per forward pass
OUTPUT_DIR="outputs/embeddings"
LIMIT=""             # Default: process all

# Parse arguments (optional overrides)
while [ "$#" -gt 0 ]; do
    case $1 in
        --model_size) MODEL_SIZE="$2"; shift ;;
        --pooling) POOLING="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p /home/haohw/vjepa2/logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/home/haohw/vjepa2/logs/extraction_${TIMESTAMP}.log"

# Check for conda environment
if command -v conda >/dev/null 2>&1; then
    # Source conda.sh if available
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . /opt/conda/etc/profile.d/conda.sh
    fi
    conda activate jepa || echo "Warning: Could not activate 'jepa' env. Proceeding with current python..."
fi

echo "========================================================"
echo "Starting V-JEPA2 Embedding Extraction (Background)"
echo "--------------------------------------------------------"
echo "Model Size:  $MODEL_SIZE"
echo "Pooling:     $POOLING"
echo "Batch Size:  $BATCH_SIZE pairs (total $((BATCH_SIZE * 2)) videos)"
if [ -n "$LIMIT" ]; then
    echo "Limit:       $LIMIT pairs per directory"
fi
echo "Output Dir:  $OUTPUT_DIR"
echo "Log File:    $LOG_FILE"
echo "========================================================"

# Build command string
CMD="python -u -m scripts.extract_embeddings \
    --model_size $MODEL_SIZE \
    --pooling $POOLING \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR \
    --device cuda"

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Run in background
# shellcheck disable=SC2086
nohup $CMD > "$LOG_FILE" 2>&1 &
PID=$!

echo "Process started with PID: $PID"
echo "Monitor logs with: tail -f $LOG_FILE"
