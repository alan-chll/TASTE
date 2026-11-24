"""Dev metrics and early stopping. Never reads test.jsonl."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from taste.rewards import r_qa_proxy
from taste.ckpt import batch_to_device
from taste.features import r_term


def needed_budget(sample: dict) -> int:
    meta = sample.get("_meta") or {}
    if not meta.get("answerable", True):
        return 0
    sem = sample.get("_answer_semantics") or "per_segment_disjunctive"
    if sem == "set_union":
        og = sample.get("_object_to_gold") or {}
        n = sum(1 for v in og.values() if v)
        return max(n, 1)
    return max(int(meta.get("M") or len(sample.get("TC") or [])), 1)


def binary_ap(y: torch.Tensor, p: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    keep = mask > 0
    yy = y[keep]
    pp = p[keep]
    n_pos = float(yy.sum())
    if yy.numel() == 0 or n_pos <= 0:
        return None
    order = torch.argsort(pp, descending=True)
    rel = yy[order]
    tp = 0.0
    ap = 0.0
    for i, r in enumerate(rel.tolist(), start=1):
        if r > 0.5:
            tp += 1.0
            ap += tp / i
    return ap / n_pos


class EarlyStop:
    def __init__(self, patience: int, mode: str = "max", min_delta: float = 1e-4):
        if mode not in ("max", "min"):
            raise ValueError(mode)
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.best_epoch = 0
        self.bad = 0

    def step(self, value: float, epoch: int) -> tuple[bool, bool]:
        """Return (is_best, should_stop). patience < 1 disables stopping but still tracks best."""
        v = float(value)
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "max":
            improved = v > self.best + self.min_delta
        else:
            improved = v < self.best - self.min_delta
        if improved:
            self.best = v
            self.best_epoch = int(epoch)
            self.bad = 0
            return True, False
        self.bad += 1
        stop = self.patience >= 1 and self.bad >= self.patience
        return False, stop


def _mean(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


@torch.no_grad()
def eval_stage0(model, loader, device, w_plus: float, w_minus: float) -> Dict[str, float]:
    """Empty-state ranking. gold_ap / prefix_rqa skip unanswerable."""
    was_train = model.training
    model.eval()
    losses = []
    aps = []
    prefix = []
    n = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        h_e, s_sem = model.encode_batch(batch["mc"], batch["texts"])
        loss = model.stage0_loss(
            h_e, s_sem, batch["tau"], batch["m_time"], batch["y"], batch["cand_mask"], w_plus, w_minus
        )
        losses.append(float(loss.item()))
        logits = model.empty_sel_logits(h_e, s_sem, batch["tau"], batch["m_time"], batch["cand_mask"])
        p = torch.sigmoid(logits)
        y = batch["y"]
        mask = batch["cand_mask"]
        for i, ex in enumerate(batch["sample"]):
            n += 1
            if not (ex.get("_meta") or {}).get("answerable", True):
                continue
            ap = binary_ap(y[i], p[i], mask[i])
            if ap is not None:
                aps.append(ap)
            scores = p[i].detach().cpu()
            msk = mask[i].detach().cpu()
            order = torch.argsort(scores + (msk - 1.0) * 1e9, descending=True).tolist()
            ids = batch["ids"][i]
            ranked = [ids[j] for j in order if msk[j] > 0 and ids[j]]
            bgt = needed_budget(ex)
            prefix.append(float(r_qa_proxy(ex, ranked[:bgt])))
    if was_train:
        model.train()
    out = {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "n": float(n),
        "n_ap": float(len(aps)),
        "n_prefix": float(len(prefix)),
    }
    apm = _mean(aps)
    prm = _mean(prefix)
    if apm is not None:
        out["gold_ap"] = apm
    if prm is not None:
        out["prefix_rqa"] = prm
    return out


@torch.no_grad()
def eval_greedy(model, loader, device, cfg: dict, t_max: int) -> Dict[str, float]:
    was_train = model.training
    model.eval()
    u_all, qa_all, qa_ans = [], [], []
    nsel_all, nsel_ans = [], []
    unans_empty = []
    first_ok = []
    n = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        roll = model.rollout(batch, greedy=True, t_max=t_max)
        for ex, ids in zip(batch["sample"], roll["selected_ids"]):
            n += 1
            st = r_term(ex, ids, cfg)
            u_all.append(float(st["u"]))
            qa_all.append(float(st["r_qa"]))
            nsel_all.append(float(len(ids)))
            ans = bool((ex.get("_meta") or {}).get("answerable", True))
            if ans:
                qa_ans.append(float(st["r_qa"]))
                nsel_ans.append(float(len(ids)))
                ok = None
                for k in range(1, len(ids) + 1):
                    if r_qa_proxy(ex, ids[:k]) == 1.0:
                        ok = k
                        break
                if ok is not None:
                    first_ok.append(float(ok))
            else:
                unans_empty.append(1.0 if not ids else 0.0)
    if was_train:
        model.train()
    out = {
        "n": float(n),
        "u": float(sum(u_all) / max(len(u_all), 1)),
        "r_qa": float(sum(qa_all) / max(len(qa_all), 1)),
        "n_sel": float(sum(nsel_all) / max(len(nsel_all), 1)),
    }
    if qa_ans:
        out["r_qa_ans"] = float(sum(qa_ans) / len(qa_ans))
        out["n_sel_ans"] = float(sum(nsel_ans) / len(nsel_ans))
    if first_ok:
        out["first_ok"] = float(sum(first_ok) / len(first_ok))
        out["n_ok"] = float(len(first_ok))
    if unans_empty:
        out["unans_empty"] = float(sum(unans_empty) / len(unans_empty))
        out["n_unans"] = float(len(unans_empty))
    return out


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fmt_metrics(d: dict) -> str:
    parts = []
    for k in (
        "loss",
        "gold_ap",
        "prefix_rqa",
        "u",
        "r_qa",
        "r_qa_ans",
        "n_sel",
        "n_sel_ans",
        "first_ok",
        "unans_empty",
    ):
        if k in d and d[k] is not None:
            parts.append(f"{k}={float(d[k]):.4f}")
    return " ".join(parts)
