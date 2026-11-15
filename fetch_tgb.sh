#!/bin/bash
set -u
ZIP="$1"
DIR="$2"
BASE="https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb"
URL="$BASE/$ZIP"
EXPECTED=$(curl -sIL --max-time 60 "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:" {v=$2} END {print v}')
if [ -z "$EXPECTED" ]; then echo "cannot determine size"; exit 1; fi
CHUNK=$((4 * 1024 * 1024))
N=$(( (EXPECTED + CHUNK - 1) / CHUNK ))
PARTDIR="/data/ljh/RKD/.download-parts/$ZIP"
mkdir -p "$PARTDIR"
rm -f "$PARTDIR"/part_*.bin
echo "zip=$ZIP dir=$DIR expected=$EXPECTED chunks=$N"
for i in $(seq 0 $((N - 1))); do
  printf "%s %s %s %s %s\n" "$ZIP" "$DIR" "$EXPECTED" "$i" "$CHUNK"
done | xargs -P 32 -n 5 bash /data/ljh/RKD/fetch_part.sh
rc=$?
echo "PARTS_EXIT=$rc"
missing=0
for i in $(seq 0 $((N - 1))); do
  lo=$((i * CHUNK)); hi=$((lo + CHUNK - 1)); [ "$hi" -ge "$EXPECTED" ] && hi=$((EXPECTED - 1))
  want=$((hi - lo + 1))
  f="$PARTDIR/part_$(printf '%05d' "$i").bin"
  have=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$have" -ne "$want" ]; then echo "chunk $i size $have != $want"; missing=$((missing + 1)); fi
done
if [ "$missing" -ne 0 ]; then echo "INCOMPLETE chunks=$missing"; exit 2; fi
mkdir -p "$DIR"
TARGET="$DIR/$ZIP"
: > "$TARGET"
for i in $(seq 0 $((N - 1))); do cat "$PARTDIR/part_$(printf '%05d' "$i").bin" >> "$TARGET"; done
actual=$(stat -c%s "$TARGET")
echo "assembled=$actual expected=$EXPECTED"
if [ "$actual" -ne "$EXPECTED" ]; then echo "SIZE_MISMATCH"; exit 3; fi
unzip -tq "$TARGET" > /dev/null 2>&1
echo "ZIP_TEST_EXIT=$?"
sha256sum "$TARGET" | tee "$DIR/zip.sha256"
unzip -oq "$TARGET" -d "$DIR"
echo "UNZIP_EXIT=$?"
rm -rf "$PARTDIR"
ls -lh "$DIR" | head -20
