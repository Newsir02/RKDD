"""Download/validate through the existing loader; record actual local versions."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import benchmark
import train


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["tgbl-wiki"])
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--output", default="results/data_manifest.json")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    records = []
    for name in args.datasets:
        options = benchmark.parse_args(["--stage", "teacher", "--dataset", name, "--data-root", args.data_root])
        options.tgb_download = args.download
        bundle, *_ = train.make_loaders(options)
        scale = benchmark.normalize_train_time(bundle)
        backend = bundle.tgb_dataset
        try:
            from tgb.utils.info import DATA_VERSION_DICT
            version = DATA_VERSION_DICT.get(bundle.dataset_name)
            package = importlib.metadata.version("py-tgb")
        except ImportError:
            version, package = None, None
        record = {"dataset": name, "backend_name": bundle.dataset_name, "dataset_version": version,
                  "py_tgb": package, "nodes": int(bundle.node_features.size(0)),
                  "events": int(bundle.data.num_events), "train": int(bundle.train_data.num_events),
                  "val": int(bundle.val_data.num_events), "test": int(bundle.test_data.num_events),
                  "metric": getattr(backend, "eval_metric", "mrr"), "official": bundle.official_evaluation,
                  "data_signature": benchmark.data_fingerprint(options, bundle), "time_scale": scale}
        if backend is not None:
            backend.load_val_ns()
            backend.load_test_ns()
        negative_paths = sorted(Path(args.data_root).rglob(f"{bundle.dataset_name}*ns.pkl"))
        record["negative_file_sha256"] = {str(p): benchmark.sha256_file(p) for p in negative_paths}
        print(json.dumps(record), flush=True)
        records.append(record)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_json(output, records)


if __name__ == "__main__":
    main()
