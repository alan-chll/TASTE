# TASTE (anonymous code release)

Time-Aware Sequential Evidence Selection with Adaptive Stopping for time-sensitive question answering.

This archive contains the TASTE method (model, features, Stage-0 / Stage-1 training, greedy inference), paper metrics, ablation configs, multi-seed scripts, and the three Full checkpoints used for the paper main results (seeds 0/1/2).

## Layout

```
TASTE_software/
  configs/default.yaml
  configs/ablations/{wo_time,wo_attn,wo_cost,wo_stage0}.yaml
  scripts/train_seeds.sh
  scripts/eval_seeds.sh
  scripts/run_ablations.sh
  checkpoints/seed{0,1,2}/stage1.pt
  requirements.txt
  taste/
```

Place the companion data archive next to this folder:

```
parent/
  TASTE_software/
  TASTE_data/
    temppool_timeqa/{train,dev,test}.jsonl
    temppool_situatedqa/test.jsonl
```

## Environment

Python 3.10+ with CUDA recommended.

```bash
pip install -r requirements.txt
```

Default encoder: `sentence-transformers/all-MiniLM-L6-v2` (downloaded on first run).

## Reproduce paper metrics (recommended)

Released Full checkpoints already match seeds 0/1/2 in the paper. With `TASTE_data/` beside this folder:

```bash
bash scripts/eval_seeds.sh
```

This writes `outputs/seed*/<dataset>/{selected,metrics}.jsonl|json` with Suff., Useful, \(F_{\mathrm{SU}}\), Conflict, Abst., and sufficient-prefix cost.

## Train from scratch

```bash
# seeds 0 1 2 (override with: bash scripts/train_seeds.sh 0)
GPU=0 bash scripts/train_seeds.sh
```

Or manually:

```bash
python -m taste.train_stage0 --config configs/default.yaml --seed 0
python -m taste.train_stage1 --config configs/default.yaml \
  --init checkpoints/seed0/stage0.pt --seed 0 --out checkpoints/seed0/stage1.pt
```

## Ablations (paper Section 5.2)

Configs under `configs/ablations/` implement:

| Config           | Paper name | Protocol                                                   |
| ---------------- | ---------- | ---------------------------------------------------------- |
| `wo_time.yaml`   | w/o time   | zero temporal features; \(\eta=\xi=0\); retrain Stage0+1   |
| `wo_attn.yaml`   | w/o attn   | skip pool self-attention; retrain Stage0+1                 |
| `wo_cost.yaml`   | w/o cost   | \(\gamma=0\); retrain Stage1 (from Full Stage0 if present) |
| `wo_stage0.yaml` | w/o Stage0 | random init, Stage1 only                                   |

```bash
SEED=0 GPU=0 bash scripts/run_ablations.sh
# or a subset:
SEED=0 bash scripts/run_ablations.sh wo_time wo_attn
```

Stage0-only (w/o Stage1) uses a Stage0 checkpoint as a ranker; see the paper for the matched-\(k\) / matched false-abstention protocols. This release focuses on sequential TASTE training and the Full/ablation Stage1 models.

## Notes

- Hyperparameters in `configs/default.yaml` match the paper.
- Baselines (Contriever, BGE, Qwen3, MRAG, DynRAG, Adapt-\(k\)) are not included.
