#!/bin/bash
set -u
cd /data/ljh/RKD
nohup bash finalize_wiki.sh > results/finalize-wiki-nohup.log 2>&1 &
echo $! > results/finalize-wiki.pid
nohup bash download_rest.sh > results/download-rest-nohup.log 2>&1 &
echo $! > results/download-rest.pid
echo "finalizer=$(cat results/finalize-wiki.pid) download_rest=$(cat results/download-rest.pid)"
