"""Offline recomputable reward pieces (except reader R_qa)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .dates import parse_bound, parse_span_list, tau_max_spans, tau_one


def parse_tc(tc_json: list):
    out = []
    for seg in tc_json:
        out.append((parse_bound(seg[0]), parse_bound(seg[1])))
    return out


def candidate_tau(cand: dict, tc) -> int:
    return tau_max_spans(parse_span_list(cand.get("time_span")), tc)


def r_ret(s_ids: Sequence[str], gold_ids: Sequence[str], answerable: bool) -> float:
    gold = set(gold_ids)
    s = set(s_ids)
    if not answerable:
        return 1.0 if not s else 0.0
    if not gold:
        return 0.0
    return len(s & gold) / len(gold)


def r_qa_proxy(sample: dict, s_ids: Sequence[str]) -> float:
    s = set(s_ids)
    meta = sample.get("_meta") or {}
    if not meta.get("answerable", True):
        return 1.0 if not s else 0.0
    sem = sample.get("_answer_semantics") or "per_segment_disjunctive"
    if sem == "set_union":
        og = sample.get("_object_to_gold") or {}
        if not og:
            return 0.0
        return float(all(s & set(v) for v in og.values()))
    m = int(meta.get("M") or len(sample.get("TC") or []))
    tg = sample.get("_tc_segment_to_gold") or {}
    for k in range(m):
        need = set(tg.get(str(k), tg.get(k, [])))
        if not (s & need):
            return 0.0
    return 1.0


def r_time(sample: dict, s_ids: Sequence[str]) -> float:
    if not s_ids:
        return 0.0
    tc = parse_tc(sample["TC"])
    by_id = {c["id"]: c for c in sample["candidates"]}
    vals = []
    for i in s_ids:
        c = by_id[i]
        t = candidate_tau(c, tc)
        vals.append(float(t))
    return sum(vals) / len(vals)


def r_cov(sample: dict, s_ids: Sequence[str]) -> float:
    meta = sample.get("_meta") or {}
    if meta.get("gold_time_case") == "B":
        return 0.0  # not counted; caller should zero xi
    tc = parse_tc(sample["TC"])
    m = len(tc)
    if m == 0:
        return 0.0
    if not meta.get("answerable", True):
        return 1.0 if not s_ids else 0.0
    by_id = {c["id"]: c for c in sample["candidates"]}
    hit = 0
    for k, seg in enumerate(tc):
        ok = False
        for i in s_ids:
            c = by_id[i]
            spans = parse_span_list(c.get("time_span"))
            if not spans:
                continue
            if any(tau_one(span, seg) == 1 for span in spans):
                ok = True
                break
        if ok:
            hit += 1
    return hit / m


def cost_count(s_ids: Sequence[str], k: int) -> float:
    return (len(s_ids) / k) if k else 0.0


def cost_tok(sample: dict, s_ids: Sequence[str], l_ctx: int) -> float:
    by_id = {c["id"]: c for c in sample["candidates"]}
    n = sum(int(by_id[i].get("_n_tokens") or 0) for i in s_ids)
    return n / l_ctx if l_ctx else 0.0
