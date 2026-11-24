"""Time features and terminal reward. Tau uses dataset multi-span overlap, not min-max envelope."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from taste.dates import parse_span_list, tau_one, tau_one_spans
from taste.rewards import cost_count, cost_tok, parse_tc, r_cov, r_qa_proxy, r_time


def candidate_tau_vec(cand: dict, tc: list, m_max: int) -> Tuple[List[float], float]:
    spans = parse_span_list(cand.get("time_span"))
    m_time = 1.0 if not spans else 0.0
    tau = [0.0] * m_max
    for k, seg in enumerate(tc[:m_max]):
        if not spans:
            tau[k] = 0.0
        else:
            tau[k] = float(tau_one_spans(spans, seg))
    return tau, m_time


def coverage_vec(sample: dict, selected_ids: Sequence[str], m_max: int) -> List[float]:
    """Empty selection -> all 0. Non-empty -> +1 / -1 on real TC segments, 0 on pad."""
    tc = parse_tc(sample.get("TC") or [])
    m = min(len(tc), m_max)
    out = [0.0] * m_max
    if not selected_ids:
        return out
    by_id = {c["id"]: c for c in sample.get("candidates") or []}
    for k in range(m):
        hit = False
        for cid in selected_ids:
            c = by_id.get(cid) or {}
            spans = parse_span_list(c.get("time_span"))
            if spans and any(tau_one(sp, tc[k]) == 1 for sp in spans):
                hit = True
                break
        out[k] = 1.0 if hit else -1.0
    return out


def r_term(sample: dict, s_ids: Sequence[str], cfg: dict) -> dict:
    meta = sample.get("_meta") or {}
    k = int(meta.get("K") or cfg.get("k") or 10)
    l_ctx = int(cfg.get("l_ctx") or 1024)
    lam = float(cfg.get("lambda_qa", 1.0))
    eta = float(cfg.get("eta_time", 0.2))
    xi = float(cfg.get("xi_cov", 0.2))
    gamma = float(cfg.get("gamma_cost", 0.3))
    if meta.get("gold_time_case") == "B":
        xi = 0.0
    qa = float(r_qa_proxy(sample, s_ids))
    tm = float(r_time(sample, s_ids))
    cv = float(r_cov(sample, s_ids))
    if (cfg.get("cost") or "count") == "tok":
        cost = float(cost_tok(sample, s_ids, l_ctx))
    else:
        cost = float(cost_count(s_ids, k))
    u = lam * qa + eta * tm + xi * cv - gamma * cost
    return {
        "u": u,
        "r_qa": qa,
        "r_time": tm,
        "r_cov": cv,
        "cost": cost,
        "xi": xi,
    }
