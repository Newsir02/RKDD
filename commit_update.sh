#!/bin/bash
set -u
cd /data/ljh/RKD
git add -A
git commit -m "Update Wiki experiment metadata and TRD ablation results" || true
git log --oneline -2
echo "files=$(git ls-files | wc -l)"
echo "tracked_pt=$(git ls-files | grep -c '\.pt$' || true)"
git count-objects -vH | tail -3
