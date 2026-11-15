#!/bin/bash
set -u
cd /data/ljh/RKD
mkdir -p .download-test
URL="https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb/tgbl-wiki-v2.zip"
rm -f .download-test/chunk* .download-test/single.bin
echo "--- tools ---"
command -v wget || true
command -v aria2c || true
echo "--- single 1MiB range ---"
start=$(date +%s.%N)
curl -L --fail -s --max-time 120 --range 0-1048575 -o .download-test/single.bin "$URL"
rc=$?
end=$(date +%s.%N)
echo "rc=$rc bytes=$(stat -c%s .download-test/single.bin 2>/dev/null || echo 0) seconds=$(echo "$end - $start" | bc)"
echo "--- 8 parallel 1MiB ranges ---"
start=$(date +%s.%N)
for i in $(seq 0 7); do
  lo=$((i * 1048576))
  hi=$((lo + 1048575))
  curl -L --fail -s --max-time 180 --range ${lo}-${hi} -o .download-test/chunk${i}.bin "$URL" &
done
wait
end=$(date +%s.%N)
total=0
for f in .download-test/chunk*.bin; do total=$((total + $(stat -c%s "$f"))); done
echo "bytes=$total seconds=$(echo "$end - $start" | bc)"
