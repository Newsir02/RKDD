#!/bin/bash
set -u
BASE="https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb"
for f in tgbl-wiki-v2.zip tgbl-review-v2.zip tgbl-coin-v2.zip tgbl-comment.zip tgbl-flight-v2.zip; do
  line=$(curl -sIL --max-time 30 "$BASE/$f" | tr -d '\r' | awk 'tolower($1)=="content-length:" {v=$2} END {print v}')
  printf "%-24s %s\n" "$f" "${line:-UNKNOWN}"
done
