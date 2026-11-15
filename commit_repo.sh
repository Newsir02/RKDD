#!/bin/bash
set -u
cd /data/ljh/RKD
git commit -m "TRD code and Wiki experiment metadata snapshot"
echo "commit_exit=$?"
git log --oneline -1
echo "untracked_or_modified=$(git status --short | wc -l)"
git count-objects -vH
