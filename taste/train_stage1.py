#!/usr/bin/env python3
"""Stage1: REINFORCE on terminal R_term. Trains Enc, MLP_sel, MLP_stop."""
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

from taste.ckpt import apply_cfg_flags, batch_to_device, load_ckpt, resolve, save_ckpt, set_seed
from taste.config import load_yaml
from taste.data import JsonlSelectDataset, collate
from taste.eval_loop import EarlyStop, append_jsonl, eval_greedy, fmt_metrics
from taste.features import r_term


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
    p.add_argument("--init", required=True, help="stage0.pt or previous stage1.pt")
    p.add_argument("--train-jsonl", default=None)
    p.add_argument("--dev-jsonl", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO / cfg_path
    cfg = load_yaml(cfg_path)
    train_path = Path(args.train_jsonl) if args.train_jsonl else resolve(cfg, "train_jsonl")
    if not train_path.is_absolute():
        train_path = REPO / train_path
    device = torch.device(args.device or cfg.get("device") or "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    init_path = Path(args.init)
    if not init_path.is_absolute():
        init_path = REPO / init_path
    model, saved_cfg = load_ckpt(init_path, map_location="cpu")
    for k, v in saved_cfg.items():
        cfg.setdefault(k, v)
    # Prefer the launched YAML for paper-aligned flags / ablation overlays.
    yaml_cfg = load_yaml(cfg_path)
    for key in (
        "mask_tau_neg",
        "skip_attn",
        "zero_time_feats",
        "eta_time",
        "xi_cov",
        "gamma_cost",
        "lambda_qa",
        "entropy_coef",
        "cost",
    ):
        if key in yaml_cfg:
            cfg[key] = yaml_cfg[key]
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    set_seed(int(cfg.get("seed") or 0))
    apply_cfg_flags(model, cfg)
    model = model.to(device)
    print(
        f"stage1 seed={cfg.get('seed')} mask_tau_neg={model.mask_tau_neg} eta_time={cfg.get('eta_time')} "
        f"xi_cov={cfg.get('xi_cov')} gamma_cost={cfg.get('gamma_cost')}",
        flush=True,
    )

    ds = JsonlSelectDataset(train_path, int(cfg.get("m_max") or 4), int(cfg.get("n") or 20), limit=args.limit)
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
        print("no dev_jsonl; will keep last epoch as stage1.pt", flush=True)

    lr = float(cfg.get("lr") or 2e-5)
    lr_enc = float(cfg.get("lr_encoder") or lr)
    opt = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr_enc},
            {
                "params": list(model.cand_fuse.parameters())
                + list(model.cand_attn.parameters())
                + list(model.cand_ln.parameters())
                + list(model.mlp_sel.parameters())
                + list(model.mlp_stop.parameters()),
                "lr": lr,
            },
        ],
        weight_decay=float(cfg.get("weight_decay") or 0.01),
    )
    epochs = int(args.epochs or cfg.get("stage1_epochs") or 12)
    patience = int(args.patience if args.patience is not None else cfg.get("stage1_patience") or 3)
    clip = float(cfg.get("grad_clip") or 1.0)
    t_max = int(cfg.get("t_max") or 11)
    ent_c = float(cfg.get("entropy_coef") or 0.0)
    min_delta = float(cfg.get("min_delta") or 1e-4)
    stopper = EarlyStop(patience, mode="max", min_delta=min_delta)

    out_best = Path(args.out) if args.out else resolve(cfg, "ckpt_dir") / "stage1.pt"
    if not out_best.is_absolute():
        out_best = REPO / out_best
    out_last = out_best.with_name(out_best.stem + "_last.pt")
    hist = out_best.with_name("stage1_history.jsonl")
    if hist.exists():
        hist.unlink()

    model.train()
    for ep in range(1, epochs + 1):
        tot_loss = 0.0
        tot_u = 0.0
        tot_qa = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"stage1 ep{ep}"):
            batch = batch_to_device(batch, device)
            roll = model.rollout(batch, greedy=False, t_max=t_max)
            stats = [r_term(ex, ids, cfg) for ex, ids in zip(batch["sample"], roll["selected_ids"])]
            u = torch.tensor([s["u"] for s in stats], dtype=torch.float32, device=device)
            qa = torch.tensor([s["r_qa"] for s in stats], dtype=torch.float32, device=device)
            base = u.mean().detach()
            logp = roll["logp"]
            ent = roll["entropy"]
            loss = -((u - base) * logp).mean() - ent_c * ent.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot_loss += float(loss.item())
            tot_u += float(u.mean().item())
            tot_qa += float(qa.mean().item())
            n += 1
        train = {
            "loss": tot_loss / max(n, 1),
            "u": tot_u / max(n, 1),
            "r_qa": tot_qa / max(n, 1),
        }
        row = {"epoch": ep, "train": train}
        save_ckpt(out_last, model, cfg, extra={"stage": 1, "epoch": ep, "kind": "last"})
        if val_loader is None:
            save_ckpt(out_best, model, cfg, extra={"stage": 1, "epoch": ep, "kind": "last_no_val"})
            print(
                f"epoch {ep} train {fmt_metrics(train)} wrote {out_best}",
                flush=True,
            )
            append_jsonl(hist, row)
            continue
        val = eval_greedy(model, val_loader, device, cfg, t_max)
        row["val"] = val
        score = float(val["u"])
        is_best, should_stop = stopper.step(score, ep)
        row["monitor"] = score
        row["best_epoch"] = stopper.best_epoch
        if is_best:
            save_ckpt(
                out_best,
                model,
                cfg,
                extra={"stage": 1, "epoch": ep, "kind": "best", "monitor": "u", "monitor_value": score, "val": val},
            )
        print(
            f"epoch {ep} train {fmt_metrics(train)} val {fmt_metrics(val)} "
            f"best_u={stopper.best:.4f}@ep{stopper.best_epoch}{' *' if is_best else ''}",
            flush=True,
        )
        append_jsonl(hist, row)
        if should_stop:
            print(
                f"early stop: no val U improve for {patience} epochs "
                f"(best {stopper.best:.4f} at epoch {stopper.best_epoch})",
                flush=True,
            )
            break
    print(f"stage1 done best={out_best} last={out_last}", flush=True)


if __name__ == "__main__":
    main()
