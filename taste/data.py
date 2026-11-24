from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from taste.rewards import parse_tc
from taste.features import candidate_tau_vec


def iter_jsonl(path: str | Path, limit: int | None = None):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if limit is not None and n >= limit:
                return


class JsonlSelectDataset(Dataset):
    def __init__(self, path: str | Path, m_max: int, n_max: int, limit: int | None = None):
        self.rows = list(iter_jsonl(path, limit=limit))
        self.m_max = int(m_max)
        self.n_max = int(n_max)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        ex = self.rows[idx]
        cands = list(ex.get("candidates") or [])[: self.n_max]
        tc = parse_tc(ex.get("TC") or [])
        gold = set(ex.get("gold_evidence_ids") or [])
        texts = []
        tau = []
        m_time = []
        y = []
        ids = []
        ntok = []
        for c in cands:
            ids.append(c["id"])
            texts.append(c.get("text") or "")
            tvec, mt = candidate_tau_vec(c, tc, self.m_max)
            tau.append(tvec)
            m_time.append(mt)
            y.append(1.0 if c["id"] in gold else 0.0)
            ntok.append(int(c.get("_n_tokens") or max(1, len((c.get("text") or "").split()))))
        n = len(cands)
        while len(texts) < self.n_max:
            ids.append("")
            texts.append("")
            tau.append([0.0] * self.m_max)
            m_time.append(0.0)
            y.append(0.0)
            ntok.append(0)
        meta = ex.get("_meta") or {}
        return {
            "sample": ex,
            "ids": ids,
            "mc": ex.get("MC") or "",
            "texts": texts,
            "tau": torch.tensor(tau, dtype=torch.float32),
            "m_time": torch.tensor(m_time, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "cand_mask": torch.tensor([1.0] * n + [0.0] * (self.n_max - n), dtype=torch.float32),
            "n_real": n,
            "k": int(meta.get("K") or 10),
            "M": int(meta.get("M") or len(tc)),
            "ntok": torch.tensor(ntok, dtype=torch.float32),
        }


def collate(batch: List[dict]) -> dict:
    out: Dict[str, object] = {
        "sample": [b["sample"] for b in batch],
        "ids": [b["ids"] for b in batch],
        "mc": [b["mc"] for b in batch],
        "texts": [b["texts"] for b in batch],
        "tau": torch.stack([b["tau"] for b in batch], dim=0),
        "m_time": torch.stack([b["m_time"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "cand_mask": torch.stack([b["cand_mask"] for b in batch], dim=0),
        "n_real": torch.tensor([b["n_real"] for b in batch], dtype=torch.long),
        "k": torch.tensor([b["k"] for b in batch], dtype=torch.float32),
        "M": torch.tensor([b["M"] for b in batch], dtype=torch.long),
        "ntok": torch.stack([b["ntok"] for b in batch], dim=0),
    }
    return out


def pos_neg_weights(rows: List[dict], n_max: int) -> tuple[float, float]:
    pos = 0
    tot = 0
    for ex in rows:
        n = min(len(ex.get("candidates") or []), n_max)
        g = len(ex.get("gold_evidence_ids") or [])
        pos += g
        tot += n
    if pos == 0 or pos == tot:
        return 1.0, 1.0
    # w+/w- = n_neg / n_pos
    w_plus = (tot - pos) / pos
    w_minus = 1.0
    return float(w_plus), float(w_minus)
