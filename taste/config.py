from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def _coerce(s: str) -> Any:
    s = s.strip()
    if s in ("null", "None", "~", ""):
        return None
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except ValueError:
        return s


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a mapping")
        return data
    except ImportError:
        root: Dict[str, Any] = {}
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip() or ":" not in line:
                continue
            key, _, rest = line.partition(":")
            root[key.strip()] = _coerce(rest.strip())
        return root
