#!/usr/bin/env python3
"""Trim a full ScanResult JSON into a compact payload for the public dashboard.

The full ScanResult of a whole-universe scan runs to several megabytes (147+
companies x 12 sub-scores x itemized evidence). GitHub Pages serves the static
dashboard, so we publish a trimmed artifact instead:

* scan metadata (generated_at, params, counts, notes) — kept verbatim, so the
  public page can state exactly what was scanned and when;
* the ``top`` list — kept in FULL (all sub-scores, evidence, narratives), these
  are the names the dashboard exists to explain;
* ``all_ranked`` — capped at ``--keep-ranked`` entries and slimmed: identity,
  headline scores, and per-category (score, confidence, coverage) triples so the
  table heat-strip still renders, but evidence text and narratives are dropped.

The trimmed file remains an honest subset: nothing is recomputed or estimated,
and a note is appended recording exactly what was trimmed.

Usage:
    python scripts/publish_scan.py data/max_scan.json frontend/latest_scan.json \
        [--keep-ranked 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List


def _slim_subscore(sub: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only what the ranked-table heat-strip and drawer bars need."""
    return {
        "category": sub.get("category"),
        "score": sub.get("score"),
        "confidence": sub.get("confidence"),
        "weight": sub.get("weight"),
        "data_coverage": sub.get("data_coverage"),
        "rationale": sub.get("rationale"),
        "evidence": [],
        "flags": sub.get("flags", []),
    }


def _slim_company(company: Dict[str, Any]) -> Dict[str, Any]:
    """Slim a non-top CompanyAnalysis: keep scores, drop narrative/evidence bulk."""
    slim = dict(company)
    slim["subscores"] = [_slim_subscore(s) for s in company.get("subscores", [])]
    # Narrative fields stay as-is if short, but non-top entries typically have
    # empty narratives already (only top-N get the explainability pass).
    return slim


def _strip_nulls(obj: Any) -> Any:
    """Recursively drop None-valued dict entries (pure size win, lossless to JS)."""
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nulls(v) for v in obj]
    return obj


def trim(scan: Dict[str, Any], keep_ranked: int) -> Dict[str, Any]:
    top: List[Dict[str, Any]] = scan.get("top", [])
    all_ranked: List[Dict[str, Any]] = scan.get("all_ranked", [])

    # Every all_ranked entry is slimmed — the dashboard overlays the full `top`
    # entries by ticker, so keeping them full here would double the payload.
    kept = [_slim_company(c) for c in all_ranked[:keep_ranked]]

    out = dict(scan)
    out["top"] = top
    out["all_ranked"] = kept
    out = _strip_nulls(out)
    dropped = max(0, len(all_ranked) - keep_ranked)
    notes = list(scan.get("notes", []))
    notes.append(
        f"Public artifact: all_ranked trimmed to the first {min(keep_ranked, len(all_ranked))} "
        f"of {len(all_ranked)} ranked names ({dropped} dropped for size); evidence text retained "
        f"for the top {len(top)} only. Nothing was recomputed — this is a subset of the full scan."
    )
    out["notes"] = notes
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="full ScanResult JSON (e.g. data/max_scan.json)")
    parser.add_argument("dst", help="output path (e.g. frontend/latest_scan.json)")
    parser.add_argument("--keep-ranked", type=int, default=100)
    args = parser.parse_args()

    with open(args.src) as fh:
        scan = json.load(fh)
    trimmed = trim(scan, args.keep_ranked)
    payload = json.dumps(trimmed, separators=(",", ":"))
    with open(args.dst, "w") as fh:
        fh.write(payload)
    size_kb = len(payload) / 1024
    print(f"wrote {args.dst}: {size_kb:,.0f} KB (top={len(trimmed['top'])}, ranked={len(trimmed['all_ranked'])})")
    if size_kb > 900:
        print("WARNING: payload approaching 1MB — consider a smaller --keep-ranked", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
