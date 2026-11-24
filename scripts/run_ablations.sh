#!/usr/bin/env bash
# Retrain paper ablations for a given seed (default 0).
# Variants: wo_time wo_attn wo_cost wo_stage0
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

SEED="${SEED:-0}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
VARIANTS="${*:-wo_time wo_attn wo_cost wo_stage0}"
FULL_STAGE0="${FULL_STAGE0:-checkpoints/seed${SEED}/stage0.pt}"

echo "== ablations seed=$SEED variants: $VARIANTS"
for V in $VARIANTS; do
  CFG="configs/ablations/${V}.yaml"
  if [[ ! -f "$CFG" ]]; then
    echo "missing $CFG" >&2
    exit 1
  fi
  OUT_DIR="checkpoints/ablations/${V}/seed${SEED}"
  mkdir -p "$OUT_DIR"

  case "$V" in
    wo_time|wo_attn)
      echo "== $(date -Is) $V stage0"
      python -m taste.train_stage0 --config "$CFG" --seed "$SEED" --out "$OUT_DIR/stage0.pt"
      echo "== $(date -Is) $V stage1"
      python -m taste.train_stage1 --config "$CFG" --seed "$SEED" --init "$OUT_DIR/stage0.pt" --out "$OUT_DIR/stage1.pt"
      ;;
    wo_cost)
      if [[ ! -f "$FULL_STAGE0" ]]; then
        echo "== $(date -Is) $V: missing Full stage0 ($FULL_STAGE0); training Stage0 with this cfg"
        python -m taste.train_stage0 --config "$CFG" --seed "$SEED" --out "$OUT_DIR/stage0.pt"
        INIT="$OUT_DIR/stage0.pt"
      else
        INIT="$FULL_STAGE0"
      fi
      echo "== $(date -Is) $V stage1 from $INIT"
      python -m taste.train_stage1 --config "$CFG" --seed "$SEED" --init "$INIT" --out "$OUT_DIR/stage1.pt"
      ;;
    wo_stage0)
      echo "== $(date -Is) $V random init then stage1"
      python - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "${ROOT}")
from taste.ckpt import build_model, save_ckpt
from taste.config import load_yaml
cfg = load_yaml("${ROOT}/${CFG}")
cfg["seed"] = int("${SEED}")
model = build_model(cfg)
out = Path("${ROOT}/${OUT_DIR}/random_init.pt")
save_ckpt(out, model, cfg, extra={"stage": 0, "kind": "random_init", "seed": int("${SEED}")})
print("wrote", out)
PY
      python -m taste.train_stage1 --config "$CFG" --seed "$SEED" --init "$OUT_DIR/random_init.pt" --out "$OUT_DIR/stage1.pt"
      ;;
    *)
      echo "unknown variant: $V" >&2
      exit 1
      ;;
  esac
done
echo "== $(date -Is) run_ablations done"
