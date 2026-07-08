#!/bin/bash
# Convexity nightly: run the full max scan and publish results to the public
# GitHub Pages site. Run by launchd (com.convexity.nightly). Idempotent; logs
# to data/nightly.log. Skips publishing when the scan is unhealthy (coverage
# gate) so a throttled run can never overwrite good public results.
set -u
ROOT="/Users/elliottwboyd/convexity"
cd "$ROOT" || exit 1
export SEC_USER_AGENT="Elliott Boyd elliottwboyd@gmail.com"
export CONVEXITY_DATA_DIR="$ROOT/data"
echo "=== nightly scan $(date) ==="
./.venv/bin/convexity scan --min-cap 10000000 --max-cap 2000000000 \
  --min-dollar-volume 100000 --top-n 25 --json "$ROOT/data/max_scan.json" || { echo "scan failed"; exit 1; }
# Coverage gate: only publish honest, near-complete data (>90% caps present).
./.venv/bin/python - <<'PY' || { echo "coverage gate FAILED — not publishing"; exit 1; }
import json, sys
d = json.load(open("/Users/elliottwboyd/convexity/data/max_scan.json"))
r = d["all_ranked"]
cov = sum(1 for c in r if c.get("market_cap")) / max(1, len(r))
print(f"coverage {cov*100:.1f}% analyzed {d['analyzed_count']} errors {d['error_count']}")
sys.exit(0 if (cov > 0.9 and d["analyzed_count"] > 1000) else 1)
PY
./.venv/bin/python scripts/publish_scan.py data/max_scan.json frontend/latest_scan.json --keep-ranked 100 || exit 1
GHP=$(mktemp -d)
git fetch origin gh-pages && git branch -f gh-pages origin/gh-pages
git worktree add "$GHP" gh-pages || exit 1
cp frontend/latest_scan.json "$GHP"/
git -C "$GHP" add -A
git -C "$GHP" commit -m "deploy: nightly scan $(date +%Y-%m-%d)" && git -C "$GHP" push origin gh-pages
git worktree remove "$GHP" --force
echo "=== nightly done $(date) ==="
