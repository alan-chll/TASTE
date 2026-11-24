from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import torch

from taste.model import HFEncoder, TASTE, TinyEncoder


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(cfg: dict, key: str, default=None):
    v = cfg.get(key, default)
    if v is None or v == "":
        return default
    p = Path(v)
    if not p.is_absolute():
        p = repo_root() / p
    return p


def build_model(cfg: dict, tiny: bool = False) -> TASTE:
    m_max = int(cfg.get("m_max") or 4)
    hidden = int(cfg.get("mlp_hidden") or 256)
    dropout = float(cfg.get("dropout") or 0.0)
    if tiny or (cfg.get("encoder") or "") == "tiny":
        enc = TinyEncoder(dim=int(cfg.get("tiny_dim") or 32))
        dim = enc.dim
    else:
        enc = HFEncoder(
            str(cfg.get("encoder") or "sentence-transformers/all-MiniLM-L6-v2"),
            int(cfg.get("max_length") or 128),
            bool(cfg.get("freeze_encoder") or False),
        )
        dim = enc.dim
    abl = cfg.get("ablation") or {}
    if not isinstance(abl, dict):
        abl = {}
    return TASTE(
        enc,
        dim,
        m_max,
        hidden,
        dropout,
        attn_heads=int(cfg.get("attn_heads") or 4),
        mask_tau_neg=bool(cfg.get("mask_tau_neg") or False),
        skip_attn=bool(abl.get("skip_attn") or cfg.get("skip_attn") or False),
        zero_time_feats=bool(abl.get("zero_time_feats") or cfg.get("zero_time_feats") or False),
    )


def apply_cfg_flags(model: TASTE, cfg: dict) -> None:
    """Runtime flags that live on the module, not in state_dict."""
    # Paper Method: no candidate is hard-filtered by temporal relevance.
    model.mask_tau_neg = bool(cfg.get("mask_tau_neg") or False)
    abl = cfg.get("ablation") or {}
    if not isinstance(abl, dict):
        abl = {}
    model.skip_attn = bool(abl.get("skip_attn") or cfg.get("skip_attn") or False)
    model.zero_time_feats = bool(abl.get("zero_time_feats") or cfg.get("zero_time_feats") or False)


def batch_to_device(batch: dict, device) -> dict:
    out = dict(batch)
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
    return out


def save_ckpt(path: Path, model: TASTE, cfg: dict, extra: Dict[str, Any] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cfg": cfg,
        "tiny": isinstance(model.encoder, TinyEncoder),
        "state_dict": model.state_dict(),
        "enc_dim": model.enc_dim,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_ckpt(path: Path, map_location="cpu") -> tuple[TASTE, dict]:
    try:
        blob = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=map_location)
    cfg = dict(blob.get("cfg") or {})
    tiny = bool(blob.get("tiny"))
    model = build_model(cfg, tiny=tiny)
    model.load_state_dict(blob["state_dict"])
    return model, cfg
