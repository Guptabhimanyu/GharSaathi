"""Income bands and which routes are open to each band. Data lives in static/scheme_rules.json (sourced, editable)."""
import json
from pathlib import Path


def _load() -> dict:
    try:
        return json.loads(Path("static/scheme_rules.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def describe(annual_income: int) -> dict:
    r = _load()
    bands = r.get("bands", [])
    hit = next((b for b in bands if annual_income <= b["max"]), None)
    band = hit["name"] if hit else None
    verts = r.get("verticals", {})
    open_v = [v for v, bs in verts.items() if band and band in bs]
    return {
        "annual": annual_income, "band": band, "band_max": hit["max"] if hit else None,
        "limits": ", ".join(f"{b['name']} up to Rs {b['max'] / 100000:g} lakh" for b in bands),
        "top": bands[-1]["max"] if bands else None, "open": open_v, "closed": [v for v in verts if v not in open_v],
        "checked": r.get("checked", ""), "known": bool(bands),
    }
