#!/usr/bin/env bash
# Greedy inference + paper metrics for seeds 0/1/2 on both test sets.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CFG="${CFG:-configs/default.yaml}"
SEEDS="${SEEDS:-0 1 2}"
DATA_ROOT="${DATA_ROOT:-../TASTE_data}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

DATASETS=(
  "temppool_timeqa:${DATA_ROOT}/temppool_timeqa/test.jsonl"
  "temppool_situatedqa:${DATA_ROOT}/temppool_situatedqa/test.jsonl"
)

echo "== eval seeds: $SEEDS"
for SEED in $SEEDS; do
  CKPT="checkpoints/seed${SEED}/stage1.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "missing $CKPT" >&2
    exit 1
  fi
  for SPEC in "${DATASETS[@]}"; do
    NAME="${SPEC%%:*}"
    JSONL="${SPEC#*:}"
    OUT_DIR="outputs/seed${SEED}/${NAME}"
    mkdir -p "$OUT_DIR"
    echo "== $(date -Is) seed=$SEED $NAME infer"
    python -m taste.infer \
      --config "$CFG" \
      --ckpt "$CKPT" \
      --jsonl "$JSONL" \
      --out "$OUT_DIR/selected.jsonl"
    echo "== $(date -Is) seed=$SEED $NAME metrics"
    python -m taste.eval_metrics \
      --jsonl "$JSONL" \
      --selected "$OUT_DIR/selected.jsonl" \
      --out "$OUT_DIR/metrics.json"
  done
done
echo "== $(date -Is) eval_seeds done"
