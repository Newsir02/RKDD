#!/bin/bash
set -u
cd /data/ljh/RKD
mkdir -p .download-test
URL="https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb/tgbl-wiki-v2.zip"
N=32
rm -f .download-test/p*.bin
start=$(date +%s.%N)
for i in $(seq 0 $((N-1))); do
  lo=$((i * 1048576))
  hi=$((lo + 1048575))
  curl -L --fail -s --max-time 300 --range ${lo}-${hi} -o .download-test/p${i}.bin "$URL" &
done
wait
end=$(date +%s.%N)
total=0
ok=0
for f in .download-test/p*.bin; do s=$(stat -c%s "$f" 2>/dev/null || echo 0); total=$((total+s)); [ "$s" -eq 1048576 ] && ok=$((ok+1)); done
echo "chunks_ok=$ok/$N bytes=$total seconds=$(echo "$end - $start" | bc)"
