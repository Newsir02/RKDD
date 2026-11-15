#!/bin/bash
set -u
cd /data/ljh/RKD
{
  echo "# TGB zip archives"
  find datasets -maxdepth 2 -name '*.zip' -print0 | sort -z | xargs -0 sha256sum
  echo "# negative sample files"
  find datasets -maxdepth 2 -name '*ns*.pkl' -print0 | sort -z | xargs -0 sha256sum
} | tee results/data-sha256.txt
