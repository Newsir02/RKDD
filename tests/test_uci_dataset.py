from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset.uci import build_uci


def test_uci_loader_sorts_events_and_builds_chronological_splits(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "uci"
    dataset_dir.mkdir()
    frame = pd.DataFrame(
        {
            "u": [5, 1, 3, 2, 4, 0, 6, 7, 8, 9],
            "i": [6, 2, 4, 3, 5, 1, 7, 8, 9, 0],
            "ts": [9, 1, 5, 2, 6, 0, 7, 8, 10, 3],
        }
    )
    frame.to_csv(dataset_dir / "ml_uci.csv", index=False)
    np.save(dataset_dir / "ml_uci.npy", np.arange(20, dtype=np.float32).reshape(10, 2))
    np.save(
        dataset_dir / "ml_uci_node.npy",
        np.arange(20, dtype=np.float32).reshape(10, 2),
    )

    bundle = build_uci(
        root=str(tmp_path),
        feature_dim=2,
        seed=7,
        num_negatives=4,
    )

    assert bool((bundle.data.t[1:] >= bundle.data.t[:-1]).all())
    assert bundle.train_data.num_events == 7
    assert bundle.val_data.num_events == 1
    assert bundle.test_data.num_events == 2
    assert bundle.node_features.shape == (10, 2)
    assert bundle.data.msg.shape == (10, 2)
    assert bundle.tgb_dataset.eval_metric == "mrr"

    sampler = bundle.tgb_dataset.negative_sampler
    negatives = sampler.query_batch(
        bundle.val_data.src,
        bundle.val_data.dst,
        bundle.val_data.t,
        split_mode="val",
    )
    assert len(negatives) == 1
    assert negatives[0].shape == (4,)
    assert int(negatives[0][0]) not in {
        int(bundle.val_data.src[0]),
        int(bundle.val_data.dst[0]),
    }


def test_uci_loader_rejects_missing_files(tmp_path: Path) -> None:
    try:
        build_uci(root=str(tmp_path))
    except FileNotFoundError as exc:
        assert "ml_uci.csv" in str(exc)
    else:
        raise AssertionError("Expected a missing-file error.")
