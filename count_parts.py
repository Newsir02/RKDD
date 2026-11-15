import os
from pathlib import Path

zip_name = "tgbl-review-v2.zip"
expected = 539628807
chunk = 4 * 1024 * 1024
n = (expected + chunk - 1) // chunk
partdir = Path("/data/ljh/RKD/.download-parts") / zip_name
missing = []
complete = 0
for i in range(n):
    lo = i * chunk
    hi = min(lo + chunk - 1, expected - 1)
    want = hi - lo + 1
    path = partdir / f"part_{i:05d}.bin"
    have = path.stat().st_size if path.exists() else 0
    if have == want:
        complete += 1
    else:
        missing.append((i, have, want))
print(f"chunks={n} complete={complete} missing={len(missing)}")
for item in missing[:50]:
    print(item)
