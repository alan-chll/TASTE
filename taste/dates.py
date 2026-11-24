"""Day-granularity closed intervals, infinity sentinels, and tau."""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from typing import List, Optional, Tuple, Union

INF_LO = date(1, 1, 1)
INF_HI = date(9999, 12, 31)
INF_LO_S = "-inf"
INF_HI_S = "+inf"

Bound = Union[date, str]


def last_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def clamp_year(y: int) -> int:
    return max(1, min(9999, y))


def parse_bound(s: Optional[str]) -> Optional[date]:
    if s is None:
        return None
    if s in (INF_LO_S, "-inf"):
        return INF_LO
    if s in (INF_HI_S, "+inf"):
        return INF_HI
    y, m, d = (int(x) for x in s.split("-"))
    return date(y, m, d)


def fmt_bound(d: Optional[date], inf_lo: bool = False, inf_hi: bool = False) -> Optional[str]:
    if d is None:
        return None
    if d <= INF_LO or inf_lo:
        if d == INF_LO:
            return INF_LO_S
    if d >= INF_HI or inf_hi:
        if d == INF_HI:
            return INF_HI_S
    if d == INF_LO:
        return INF_LO_S
    if d == INF_HI:
        return INF_HI_S
    return d.isoformat()


def to_json_span(span: Optional[Tuple[date, date]]) -> Optional[list]:
    if span is None:
        return None
    a, b = span
    return [fmt_bound(a), fmt_bound(b)]


def as_span_list(ts) -> List[list]:
    """Normalize JSON `time_span` to a list of [lo, hi] string pairs.

    Current format is a list of pairs. Legacy format is a single pair.
    Malformed values yield [].
    """
    if not ts:
        return []
    first = ts[0]
    if isinstance(first, (list, tuple)):
        out = []
        for p in ts:
            if not p or len(p) != 2:
                return []
            if not isinstance(p[0], str) or not isinstance(p[1], str):
                return []
            out.append([p[0], p[1]])
        return out
    if (
        isinstance(first, str)
        and len(ts) == 2
        and isinstance(ts[1], str)
        and not isinstance(ts[1], list)
    ):
        return [[ts[0], ts[1]]]
    return []


def parse_span_list(ts) -> List[Tuple[date, date]]:
    out: List[Tuple[date, date]] = []
    for pair in as_span_list(ts):
        try:
            a, b = parse_bound(pair[0]), parse_bound(pair[1])
        except (TypeError, ValueError):
            continue
        if a is None or b is None or a > b:
            continue
        out.append((a, b))
    return out


def to_json_spans(spans: Optional[List[Tuple[date, date]]]) -> Optional[list]:
    if not spans:
        return None
    out = []
    for sp in spans:
        js = to_json_span(sp)
        if js:
            out.append(js)
    return out or None


def day_before(d: date) -> date:
    if d <= INF_LO:
        return INF_LO
    return d - timedelta(days=1)


def day_after(d: date) -> date:
    if d >= INF_HI:
        return INF_HI
    return d + timedelta(days=1)


def start_of_year(y: int) -> date:
    return date(clamp_year(y), 1, 1)


def end_of_year(y: int) -> date:
    return date(clamp_year(y), 12, 31)


def start_of_month(y: int, m: int) -> date:
    return date(clamp_year(y), m, 1)


def end_of_month(y: int, m: int) -> date:
    return date(clamp_year(y), m, last_day(clamp_year(y), m))


MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def parse_timeqa_point(s: str) -> date:
    """Parse TempPool-TimeQA axis labels like 'May 1997' or '1997' to a representative date (start)."""
    s = s.strip()
    if " " not in s:
        return start_of_year(int(s))
    mon, year = s.split()
    m = MONTHS[mon.lower()]
    return start_of_month(int(year), m)


def parse_timeqa_end(s: str) -> date:
    s = s.strip()
    if " " not in s:
        return end_of_year(int(s))
    mon, year = s.split()
    m = MONTHS[mon.lower()]
    return end_of_month(int(year), m)


def point_range(s: str) -> Tuple[date, date]:
    """Inclusive range of a TempPool-TimeQA time label."""
    s = s.strip()
    if " " not in s:
        y = int(s)
        return start_of_year(y), end_of_year(y)
    mon, year = s.split()
    m = MONTHS[mon.lower()]
    y = int(year)
    return start_of_month(y, m), end_of_month(y, m)


def tau_one(span: Optional[Tuple[date, date]], tc_seg: Tuple[date, date]) -> int:
    """Closed-interval overlap: +1 overlap, -1 no overlap. Missing span is handled by caller as 0."""
    if span is None:
        return 0
    a, b = span
    x1, x2 = tc_seg
    left = max(a, x1)
    right = min(b, x2)
    return 1 if (right - left).days >= 0 else -1


def tau_max(span: Optional[Tuple[date, date]], tc: list) -> int:
    if span is None:
        return 0
    vals = [tau_one(span, seg) for seg in tc]
    return max(vals) if vals else 0


def tau_one_spans(spans: Optional[List[Tuple[date, date]]], tc_seg: Tuple[date, date]) -> int:
    if not spans:
        return 0
    if any(tau_one(sp, tc_seg) == 1 for sp in spans):
        return 1
    return -1


def tau_max_spans(spans: Optional[List[Tuple[date, date]]], tc: list) -> int:
    if not spans:
        return 0
    vals = [tau_one_spans(spans, seg) for seg in tc]
    return max(vals) if vals else 0


def spans_overlap(a: Tuple[date, date], b: Tuple[date, date]) -> bool:
    return tau_one(a, b) == 1


_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")


def years_in_text(text: str) -> list:
    return [int(y) for y in _YEAR_RE.findall(text or "")]
