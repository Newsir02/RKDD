from __future__ import annotations

import argparse
import json
from pathlib import Path

import benchmark


def build_jobs(config, output_root):
    root = Path(output_root)
    jobs = []
    datasets, teachers, students, seeds = (config[k] for k in ("datasets", "teachers", "students", "seeds"))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds are not allowed")
    if any(m == "supervised" or m not in benchmark.METHODS for m in config["methods"]):
        raise ValueError("methods must contain distillation objectives only")
    for dataset in datasets:
        for seed in seeds:
            common = {**config["common"], "dataset": dataset, "seed": seed}
            for teacher in teachers:
                directory = root / dataset / f"seed-{seed}" / f"teacher-{teacher}"
                jobs.append({**common, "stage": "teacher", "method": "logits", "teacher_arch": teacher,
                             "output_dir": str(directory)})
            for student in students:
                directory = root / dataset / f"seed-{seed}" / student / "supervised"
                # Supervised runs are independent of the teacher; avoid triplication.
                jobs.append({**common, "stage": "student", "method": "supervised", "student_model": student,
                             "teacher_arch": teachers[0], "output_dir": str(directory)})
                for teacher in teachers:
                    checkpoint = root / dataset / f"seed-{seed}" / f"teacher-{teacher}" / "teacher.pt"
                    for method in config["methods"]:
                        directory = root / dataset / f"seed-{seed}" / student / teacher / method
                        jobs.append({**common, "stage": "distill", "method": method, "student_model": student,
                                     "teacher_arch": teacher, "teacher_checkpoint": str(checkpoint),
                                     "output_dir": str(directory)})
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/comparison.json")
    parser.add_argument("--output-root", default="results/comparison")
    parser.add_argument("--execute", action="store_true", help="Without this option, generate the manifest only")
    parser.add_argument("--resume", action="store_true", help="Skip complete runs only when config and teacher bytes match")
    parser.add_argument("--max-runs", type=int, default=0)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    jobs = build_jobs(config, args.output_root)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    benchmark.write_json(root / "manifest.json", {"jobs": jobs, "seeds": config["seeds"], "config": config})
    print(f"Planned {len(jobs)} runs; execute={args.execute}", flush=True)
    if not args.execute:
        return jobs
    completed = 0
    for number, job in enumerate(jobs):
        if args.max_runs and completed >= args.max_runs:
            break
        directory = Path(job["output_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        job_config = directory / "job.json"
        benchmark.write_json(job_config, job)
        effective = benchmark.parse_args(["--config", str(job_config)])
        result_file = directory / "result.json"
        if args.resume and result_file.exists():
            result = json.loads(result_file.read_text(encoding="utf-8"))
            if result["config_hash"] != benchmark.fingerprint(vars(effective)):
                raise ValueError(f"Cannot resume changed config: {directory}")
            if result["source_hash"] != benchmark.source_fingerprint():
                raise ValueError(f"Cannot resume after source changes: {directory}")
            model_file = directory / ("teacher.pt" if job["stage"] == "teacher" else "student.pt")
            if benchmark.sha256_file(model_file) != result["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint bytes changed: {directory}")
            if job.get("teacher_checkpoint") and benchmark.sha256_file(job["teacher_checkpoint"]) != result["teacher_checkpoint_sha256"]:
                raise ValueError(f"Shared teacher bytes changed: {directory}")
            continue
        print(f"Run {number + 1}/{len(jobs)}: {directory}", flush=True)
        try:
            benchmark.run(effective)
        except Exception as exc:
            benchmark.write_json(directory / "failure.json", {"status": "failed", "error": repr(exc)})
            raise
        completed += 1
    return jobs


if __name__ == "__main__":
    main()
