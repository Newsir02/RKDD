import argparse
import hashlib
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = "https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb"
CHUNK = 4 * 1024 * 1024


def expected_size(url):
    for attempt in range(5):
        try:
            r = requests.head(url, timeout=60, allow_redirects=True)
            r.raise_for_status()
            return int(r.headers["content-length"])
        except Exception as exc:
            print(f"HEAD attempt {attempt + 1} failed: {exc}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"cannot determine size for {url}")


def fetch_part(url, part, lo, hi, want):
    for attempt in range(1, 41):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return 0
        if have > want:
            part.unlink()
            have = 0
        headers = {"Range": f"bytes={lo + have}-{hi}"}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 600)) as r:
                if r.status_code != 206:
                    if part.exists():
                        part.unlink()
                    raise RuntimeError(f"expected HTTP 206, got {r.status_code}")
                mode = "ab" if have else "wb"
                remaining = want - have
                with part.open(mode) as handle:
                    for block in r.iter_content(chunk_size=256 * 1024):
                        if not block:
                            continue
                        if len(block) > remaining:
                            block = block[:remaining]
                        handle.write(block)
                        remaining -= len(block)
                        if remaining == 0:
                            break
        except Exception as exc:
            if attempt % 5 == 0 or attempt == 1:
                print(f"  retry part={part.name} attempt={attempt} have={have} err={exc}", flush=True)
            time.sleep(2)
            continue
        if part.stat().st_size == want:
            return 0
    return 1


def download(zip_name, target_dir, jobs):
    url = f"{BASE}/{zip_name}"
    expected = expected_size(url)
    n = (expected + CHUNK - 1) // CHUNK
    partdir = Path("/data/ljh/RKD/.download-parts") / zip_name
    partdir.mkdir(parents=True, exist_ok=True)
    print(f"{zip_name}: expected={expected} chunks={n} jobs={jobs}", flush=True)
    work = []
    for i in range(n):
        lo = i * CHUNK
        hi = min(lo + CHUNK - 1, expected - 1)
        want = hi - lo + 1
        part = partdir / f"part_{i:05d}.bin"
        have = part.stat().st_size if part.exists() else 0
        if have != want:
            work.append((i, part, lo, hi, want, have))
    print(f"{zip_name}: complete={n - len(work)} pending={len(work)}", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(fetch_part, url, p, lo, hi, want): i for i, p, lo, hi, want, _ in work}
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            done += 1
            if future.result() != 0:
                failed.append(idx)
            if done % 10 == 0 or done == len(work):
                print(f"{zip_name}: parts {done}/{len(work)} done, failed={len(failed)}", flush=True)
    if failed:
        print(f"{zip_name}: FAILED parts {failed}", flush=True)
        return False
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / zip_name
    with target.open("wb") as out:
        for i in range(n):
            with (partdir / f"part_{i:05d}.bin").open("rb") as handle:
                shutil.copyfileobj(handle, out)
    actual = target.stat().st_size
    print(f"{zip_name}: assembled={actual} expected={expected}", flush=True)
    if actual != expected:
        return False
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad is not None:
            print(f"{zip_name}: corrupt member {bad}", flush=True)
            return False
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (target_dir / "zip.sha256").write_text(f"{digest}  {target}\n", encoding="utf-8")
    print(f"{zip_name}: sha256={digest}", flush=True)
    with zipfile.ZipFile(target) as archive:
        archive.extractall(target_dir)
    print(f"{zip_name}: extracted to {target_dir}", flush=True)
    shutil.rmtree(partdir, ignore_errors=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_name")
    parser.add_argument("target_dir")
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    ok = download(args.zip_name, args.target_dir, args.jobs)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
