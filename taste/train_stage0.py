#!/usr/bin/env python3
"""Stage0: empty-state weighted BCE on gold labels. Trains Enc, W, Attn, LN and MLP_sel."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from taste.ckpt import batch_to_device, build_model, resolve, save_ckpt, set_seed
from taste.config import load_yaml
from taste.data import JsonlSelectDataset, collate, pos_neg_weights
from taste.eval_loop import EarlyStop, append_jsonl, eval_stage0, fmt_metrics


def _jsonl_path(raw, cfg, key):
    if raw:
        p = Path(raw)
    else:
        p = resolve(cfg, key)
    if p is None:
        return None
    if not p.is_absolute():
        p = REPO / p
    return p if p.exists() else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--train-jsonl", default=None)
    p.add_argument("--dev-jsonl", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    cfg = load_yaml(cfg_path)
    if args.tiny:
        cfg["encoder"] = "tiny"
    train_path = Path(args.train_jsonl) if args.train_jsonl else resolve(cfg, "train_jsonl")
    if not train_path.is_absolute():
        train_path = REPO / train_path
    device = torch.device(args.device or cfg.get("device") or "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    seed = int(cfg.get("seed") or 0)
    set_seed(seed)
    print(f"stage0 seed={seed}", flush=True)

    ds = JsonlSelectDataset(train_path, int(cfg.get("m_max") or 4), int(cfg.get("n") or 20), limit=args.limit)
    w_plus, w_minus = pos_neg_weights(ds.rows, int(cfg.get("n") or 20))
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size") or 8), shuffle=True, collate_fn=collate)
    if args.dev_jsonl:
        val_path = _jsonl_path(args.dev_jsonl, cfg, "dev_jsonl")
    elif args.train_jsonl:
        val_path = None
    else:
        val_path = _jsonl_path(None, cfg, "dev_jsonl")
    val_loader = None
    if val_path is not None:
        val_ds = JsonlSelectDataset(
            val_path, int(cfg.get("m_max") or 4), int(cfg.get("n") or 20), limit=args.val_limit
        )
        val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch_size") or 8), shuffle=False, collate_fn=collate)
        print(f"val {val_path} n={len(val_ds)}", flush=True)
    else:
        print("no dev_jsonl; will keep last epoch as stage0.pt", flush=True)

    model = build_model(cfg, tiny=args.tiny).to(device)
    head = list(model.cand_fuse.parameters()) + list(model.cand_attn.parameters())
    head += list(model.cand_ln.parameters()) + list(model.mlp_sel.parameters())
    params = list(model.encoder.parameters()) + head
    enc_params = [x for x in model.encoder.parameters() if x.requires_grad]
    groups = [{"params": head, "lr": float(cfg.get("lr") or 2e-5)}]
    if enc_params:
        groups.insert(0, {"params": enc_params, "lr": float(cfg.get("lr_encoder") or cfg.get("lr") or 2e-5)})
    opt = torch.optim.AdamW(groups, weight_decay=float(cfg.get("weight_decay") or 0.01))
    epochs = int(args.epochs or cfg.get("stage0_epochs") or 20)
    patience = int(args.patience if args.patience is not None else cfg.get("stage0_patience") or 4)
    clip = float(cfg.get("grad_clip") or 1.0)
    min_delta = float(cfg.get("min_delta") or 1e-4)
    stopper = EarlyStop(patience, mode="max", min_delta=min_delta)

    out_best = Path(args.out) if args.out else resolve(cfg, "ckpt_dir") / "stage0.pt"
    if not out_best.is_absolute():
        out_best = REPO / out_best
    out_last = out_best.with_name(out_best.stem + "_last.pt")
    hist = out_best.with_name("stage0_history.jsonl")
    if hist.exists():
        hist.unlink()
    extra = {"w_plus": w_plus, "w_minus": w_minus, "stage": 0}

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"stage0 ep{ep}"):
            batch = batch_to_device(batch, device)
            h_e, s_sem = model.encode_batch(batch["mc"], batch["texts"])
            loss = model.stage0_loss(
                h_e, s_sem, batch["tau"], batch["m_time"], batch["y"], batch["cand_mask"], w_plus, w_minus
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            total += float(loss.item())
            n += 1
        train_loss = total / max(n, 1)
        row = {"epoch": ep, "train_loss": train_loss, "w_plus": w_plus, "w_minus": w_minus}
        save_ckpt(out_last, model, cfg, extra={**extra, "epoch": ep, "kind": "last"})
        if val_loader is None:
            save_ckpt(out_best, model, cfg, extra={**extra, "epoch": ep, "kind": "last_no_val"})
            print(f"epoch {ep} train_loss={train_loss:.4f} wrote {out_best}", flush=True)
            append_jsonl(hist, row)
            continue
        val = eval_stage0(model, val_loader, device, w_plus, w_minus)
        row["val"] = val
        score = float(val.get("gold_ap") if val.get("gold_ap") is not None else -val["loss"])
        is_best, should_stop = stopper.step(score, ep)
        row["monitor"] = score
        row["best_epoch"] = stopper.best_epoch
        if is_best:
            save_ckpt(
                out_best,
                model,
                cfg,
                extra={**extra, "epoch": ep, "kind": "best", "monitor": "gold_ap", "monitor_value": score, "val": val},
            )
        print(
            f"epoch {ep} train_loss={train_loss:.4f} val {fmt_metrics(val)} "
            f"best={stopper.best:.4f}@ep{stopper.best_epoch}{' *' if is_best else ''}",
            flush=True,
        )
        append_jsonl(hist, row)
        if should_stop:
            print(
                f"early stop: no val gold_ap improve for {patience} epochs "
                f"(best {stopper.best:.4f} at epoch {stopper.best_epoch})",
                flush=True,
            )
            break
    print(f"stage0 done best={out_best} last={out_last}", flush=True)


if __name__ == "__main__":
    main()
