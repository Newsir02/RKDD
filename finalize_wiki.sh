#!/bin/bash
set -u
cd /data/ljh/RKD
PID=$(cat results/main-wiki.pid)
echo "watching matrix pid=$PID at $(date -Is)" >> results/finalize-wiki.log
while ps -p "$PID" > /dev/null 2>&1; do
  sleep 300
done
echo "matrix process ended at $(date -Is)" >> results/finalize-wiki.log
sleep 10
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
/home/ubuntu/anaconda3/envs/Graph/bin/python -m experiments.summarize \
  results/main --output results/main/summary.csv >> results/finalize-wiki.log 2>&1
echo "summarize exit=$?" >> results/finalize-wiki.log
/home/ubuntu/anaconda3/envs/Graph/bin/python list_results.py > results/main/all-results.txt 2>&1
echo "results=$(find results/main -name result.json | wc -l)/141" > results/WIKI_DONE.txt
echo "failures=$(find results/main -name failure.json | wc -l)" >> results/WIKI_DONE.txt
echo "finished_at=$(date -Is)" >> results/WIKI_DONE.txt
head -3 results/main/summary.csv >> results/WIKI_DONE.txt 2>&1
echo "DONE" >> results/finalize-wiki.log
