import os
from pathlib import Path

for zip_name, expected in [
    ("tgbl-coin-v2.zip", 1284466067),
    ("tgbl-comment.zip", 1710634236),
    ("tgbl-flight-v2.zip", 1281890984),
]:
    chunk = 4 * 1024 * 1024
    n = (expected + chunk - 1) // chunk
    partdir = Path("/data/ljh/RKD/.download-parts") / zip_name
    if not partdir.exists():
        print(zip_name, "not started")
        continue
    complete = 0
    total = 0
    last = []
    for i in range(n):
        lo = i * chunk
        hi = min(lo + chunk - 1, expected - 1)
        want = hi - lo + 1
        p = partdir / f"part_{i:05d}.bin"
        have = p.stat().st_size if p.exists() else 0
        total += have
        if have == want:
            complete += 1
        if i >= n - 10:
            last.append((i, have, want))
    print(zip_name, f"complete={complete}/{n} bytes={total}/{expected}")
    print("  last:", last)
