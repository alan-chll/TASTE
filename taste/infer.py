#!/usr/bin/env python3
"""Greedy sequential selection. Writes selected evidence IDs in selection order."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from taste.ckpt import apply_cfg_flags, batch_to_device, load_ckpt, resolve
from taste.config import load_yaml
from taste.data import JsonlSelectDataset, collate


def infer_loader(model, loader, device, t_max: int):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            roll = model.rollout(batch, greedy=True, t_max=t_max)
            for ex, ids in zip(batch["sample"], roll["selected_ids"]):
                rows.append(
                    {
                        "id": ex["id"],
                        "ranked_ids": ids,
                        "selected_ids": ids,
                        "selector": "taste",
                    }
                )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / args.config
    cfg = load_yaml(cfg_path)
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    jsonl = Path(args.jsonl)
    if not jsonl.is_absolute():
        jsonl = REPO / jsonl
    device = torch.device(args.device or cfg.get("device") or "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model, saved = load_ckpt(ckpt, map_location="cpu")
    for k, v in saved.items():
        cfg.setdefault(k, v)
    # Prefer YAML for paper-aligned runtime flags (no temporal hard filter).
    yaml_cfg = load_yaml(cfg_path)
    for key in ("mask_tau_neg", "skip_attn", "zero_time_feats", "eta_time", "xi_cov", "gamma_cost"):
        if key in yaml_cfg:
            cfg[key] = yaml_cfg[key]
    apply_cfg_flags(model, cfg)
    model = model.to(device)
    ds = JsonlSelectDataset(jsonl, int(cfg.get("m_max") or 4), int(cfg.get("n") or 20), limit=args.limit)
    bs = int(args.batch_size or cfg.get("batch_size") or 8)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate)
    rows = infer_loader(model, loader, device, int(cfg.get("t_max") or 11))
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
