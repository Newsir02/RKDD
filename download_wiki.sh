#!/bin/bash
set -u
cd /data/ljh/RKD
URL="https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb/tgbl-wiki-v2.zip"
DIR=/data/ljh/RKD/datasets/tgbl_wiki
mkdir -p "$DIR"
rm -f "$DIR/tgbl-wiki.zip"
echo "downloading $URL"
curl -L --fail --retry 8 --retry-delay 3 \
  --connect-timeout 30 --speed-time 60 --speed-limit 1024 \
  -o "$DIR/tgbl-wiki.zip" "$URL"
echo "CURL_EXIT=$?"
ls -lh "$DIR/tgbl-wiki.zip"
unzip -tq "$DIR/tgbl-wiki.zip" > /dev/null 2>&1
echo "ZIP_TEST_EXIT=$?"
sha256sum "$DIR/tgbl-wiki.zip"
unzip -oq "$DIR/tgbl-wiki.zip" -d "$DIR"
echo "UNZIP_EXIT=$?"
ls -lh "$DIR" | head -20
