#!/bin/bash
set -u
ZIP="$1"
DIR="$2"
EXPECTED="$3"
IDX="$4"
CHUNK="$5"
BASE="https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb"
URL="$BASE/$ZIP"
PARTDIR="/data/ljh/RKD/.download-parts/$ZIP"
lo=$((IDX * CHUNK))
hi=$((lo + CHUNK - 1))
if [ "$hi" -ge "$EXPECTED" ]; then hi=$((EXPECTED - 1)); fi
want=$((hi - lo + 1))
out="$PARTDIR/part_$(printf '%05d' "$IDX").bin"
for attempt in $(seq 1 8); do
  have=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$have" -eq "$want" ]; then exit 0; fi
  curl -L --fail -s --max-time 900 --range "${lo}-${hi}" -o "$out" "$URL"
  have=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$have" -eq "$want" ]; then exit 0; fi
  echo "retry idx=$IDX attempt=$attempt have=$have want=$want"
  sleep 2
done
exit 1
