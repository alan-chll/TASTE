#!/usr/bin/env python3
"""Paper metrics: Suff., Useful, F_SU, Conflict, Abst., false abstention, sufficient-prefix cost."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from taste.dates import parse_span_list, tau_max_spans
from taste.rewards import parse_tc, r_qa_proxy


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_selected(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for row in iter_jsonl(path):
        qid = row.get("id")
        if qid is None:
            continue
        ids = row.get("selected_ids")
        if ids is None:
            ids = row.get("ranked_ids") or []
        out[str(qid)] = list(ids)
    return out


def is_answerable(sample: dict) -> bool:
    return bool((sample.get("_meta") or {}).get("answerable", True))


def required_groups(sample: dict) -> List[Set[str]]:
    if not is_answerable(sample):
        return []
    sem = sample.get("_answer_semantics") or "per_segment_disjunctive"
    if sem == "set_union":
        return [set(v or []) for v in (sample.get("_object_to_gold") or {}).values() if v]
    meta = sample.get("_meta") or {}
    m = int(meta.get("M") or len(sample.get("TC") or []))
    tg = sample.get("_tc_segment_to_gold") or {}
    return [set(tg.get(str(k), tg.get(k, [])) or []) for k in range(m)]


def n_useful(sample: dict, ids: Sequence[str]) -> int:
    groups = required_groups(sample)
    if not groups:
        return 0
    uncovered = set(range(len(groups)))
    n = 0
    for cid in ids:
        hit = {j for j in uncovered if cid in groups[j]}
        if hit:
            n += 1
            uncovered -= hit
        if not uncovered:
            break
    return n


def n_tokens(sample: dict, ids: Sequence[str]) -> int:
    by_id = {c["id"]: c for c in sample.get("candidates") or []}
    tot = 0
    for i in ids:
        c = by_id.get(i) or {}
        if c.get("_n_tokens") is not None:
            tot += int(c["_n_tokens"])
        else:
            tot += max(1, len((c.get("text") or "").split()))
    return tot


def first_ok(sample: dict, ids: Sequence[str]) -> Optional[int]:
    if not is_answerable(sample):
        return 0 if not ids else None
    for k in range(1, len(ids) + 1):
        if r_qa_proxy(sample, ids[:k]) == 1.0:
            return k
    return None


def candidate_tau_max(sample: dict, cid: str) -> int:
    tc = parse_tc(sample.get("TC") or [])
    by_id = {c["id"]: c for c in sample.get("candidates") or []}
    c = by_id.get(cid) or {}
    return int(tau_max_spans(parse_span_list(c.get("time_span")), tc))


def f_su(suff: Optional[float], useful: Optional[float]) -> Optional[float]:
    if suff is None or useful is None:
        return None
    if suff == 0.0 and useful == 0.0:
        return 0.0
    denom = suff + useful
    if denom <= 0:
        return 0.0
    return 2.0 * suff * useful / denom


def mean(xs: Sequence[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def evaluate(samples: Iterable[dict], selected: Dict[str, List[str]]) -> dict:
    rows = []
    n_conflict_items = 0
    n_items = 0
    for sample in samples:
        qid = str(sample["id"])
        ids = list(selected.get(qid) or [])
        ans = is_answerable(sample)
        for cid in ids:
            n_items += 1
            if candidate_tau_max(sample, cid) == -1:
                n_conflict_items += 1
        row = {
            "id": qid,
            "answerable": ans,
            "n_pack": len(ids),
            "empty": len(ids) == 0,
            "r_qa": None,
            "useful": None,
            "depth": None,
            "tokens_at_ok": None,
        }
        if ans:
            row["r_qa"] = float(r_qa_proxy(sample, ids))
            row["useful"] = (n_useful(sample, ids) / len(ids)) if ids else 0.0
            d = first_ok(sample, ids)
            row["depth"] = d
            if d is not None:
                row["tokens_at_ok"] = float(n_tokens(sample, ids[:d]))
        rows.append(row)

    ans_rows = [r for r in rows if r["answerable"]]
    un_rows = [r for r in rows if not r["answerable"]]
    suff = mean([r["r_qa"] for r in ans_rows if r["r_qa"] is not None])
    useful = mean([r["useful"] for r in ans_rows if r["useful"] is not None])
    abst = mean([1.0 if r["empty"] else 0.0 for r in un_rows])
    false_abst = mean([1.0 if r["empty"] else 0.0 for r in ans_rows])
    ok = [r for r in ans_rows if r.get("r_qa") == 1.0]
    out = {
        "n": len(rows),
        "n_answerable": len(ans_rows),
        "n_unanswerable": len(un_rows),
        "sufficiency": suff,
        "useful": useful,
        "f_su": f_su(suff, useful),
        "conflict": (n_conflict_items / n_items) if n_items else 0.0,
        "n_selected_items": n_items,
        "abstention": abst,
        "false_abstention": false_abst,
        "mean_pack": mean([float(r["n_pack"]) for r in ans_rows]),
        "sufficient_prefix_items": mean([float(r["depth"]) for r in ok if r.get("depth") is not None]),
        "sufficient_prefix_tokens": mean(
            [float(r["tokens_at_ok"]) for r in ok if r.get("tokens_at_ok") is not None]
        ),
        "n_sufficient": len(ok),
    }
    return out


def fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{100.0 * float(x):.1f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", required=True, help="dataset jsonl")
    p.add_argument("--selected", required=True, help="infer output jsonl")
    p.add_argument("--out", default=None, help="optional metrics json")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    samples = []
    for i, row in enumerate(iter_jsonl(Path(args.jsonl))):
        samples.append(row)
        if args.limit is not None and i + 1 >= args.limit:
            break
    selected = load_selected(Path(args.selected))
    metrics = evaluate(samples, selected)
    print(
        f"Suff={fmt_pct(metrics['sufficiency'])} "
        f"Useful={fmt_pct(metrics['useful'])} "
        f"F_SU={fmt_pct(metrics['f_su'])} "
        f"Conflict={fmt_pct(metrics['conflict'])} "
        f"Abst={fmt_pct(metrics['abstention'])} "
        f"FalseAbst={fmt_pct(metrics['false_abstention'])} "
        f"|S|={metrics['mean_pack']:.2f}" if metrics["mean_pack"] is not None else "",
        flush=True,
    )
    if metrics.get("sufficient_prefix_items") is not None:
        print(
            f"sufficient-prefix items={metrics['sufficient_prefix_items']:.2f} "
            f"tokens={metrics['sufficient_prefix_tokens']:.1f} "
            f"(n_ok={metrics['n_sufficient']})",
            flush=True,
        )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
