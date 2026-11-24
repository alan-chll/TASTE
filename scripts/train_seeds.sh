#!/usr/bin/env bash
# Train Full TASTE for one or more seeds (paper uses 0 1 2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

CFG="${CFG:-configs/default.yaml}"
SEEDS="${*:-0 1 2}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "== train seeds: $SEEDS  cfg=$CFG  gpu=$GPU"
for SEED in $SEEDS; do
  OUT_DIR="checkpoints/seed${SEED}"
  mkdir -p "$OUT_DIR"
  echo "== $(date -Is) seed=$SEED stage0"
  python -m taste.train_stage0 \
    --config "$CFG" \
    --seed "$SEED" \
    --out "$OUT_DIR/stage0.pt"
  echo "== $(date -Is) seed=$SEED stage1"
  python -m taste.train_stage1 \
    --config "$CFG" \
    --seed "$SEED" \
    --init "$OUT_DIR/stage0.pt" \
    --out "$OUT_DIR/stage1.pt"
done
echo "== $(date -Is) train_seeds done"
