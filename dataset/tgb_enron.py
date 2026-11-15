from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import TemporalData

from .temporal import TemporalGraphBundle, build_temporal_graph_bundle


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_negative_dict(path: Path) -> Dict[tuple[int, int, int], np.ndarray]:
    with path.open("rb") as handle:
        payload = _CompatUnpickler(handle).load()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected enron negative file format: {path}")
    return {
        (int(key[0]), int(key[1]), int(key[2])): np.asarray(value, dtype=np.int64)
        for key, value in payload.items()
    }


class _EnronNegativeSampler:
    def __init__(
        self,
        val_negatives: Dict[tuple[int, int, int], np.ndarray],
        test_negatives: Dict[tuple[int, int, int], np.ndarray],
    ) -> None:
        self.val_negatives = val_negatives
        self.test_negatives = test_negatives

    def query_batch(
        self,
        src: Iterable[int],
        dst: Iterable[int],
        timestamps: Iterable[int],
        split_mode: str,
    ) -> list[np.ndarray]:
        table = (
            self.val_negatives if split_mode == "val" else self.test_negatives
        )
        result = []
        for source, destination, timestamp in zip(src, dst, timestamps):
            key = (int(source), int(destination), int(timestamp))
            if key not in table:
                raise KeyError(f"Missing enron negative samples for event {key}.")
            result.append(table[key])
        return result


class _EnronBackend:
    eval_metric = "mrr"

    def __init__(
        self,
        val_negatives: Dict[tuple[int, int, int], np.ndarray],
        test_negatives: Dict[tuple[int, int, int], np.ndarray],
    ) -> None:
        self.negative_sampler = _EnronNegativeSampler(val_negatives, test_negatives)

    def load_val_ns(self) -> None:
        return None

    def load_test_ns(self) -> None:
        return None


def build_tgb_enron(
    *,
    name: str = "tgbl-enron",
    root: str = "datasets",
    feature_dim: int = 16,
    edge_feature_dim: int = 16,
    num_candidates: int = 16,
    history_size: int = 24,
    seed: int = 42,
    max_events: int = 0,
    download: bool = True,
    structure_feature_set: str = "basic",
) -> TemporalGraphBundle:
    del edge_feature_dim, num_candidates, history_size, seed, download
    if name != "tgbl-enron":
        raise ValueError("build_tgb_enron only supports 'tgbl-enron'.")

    try:
        import tgb.utils.info as info
    except ImportError as exc:
        raise ImportError("Loading tgbl-enron requires py-tgb.") from exc
    info.DATA_EVAL_METRIC_DICT.setdefault("tgbl-enron", "mrr")

    dataset_dir = Path(root) / "tgbl_enron"
    edge_path = dataset_dir / "tgbl-enron_edgelist.csv"
    if not edge_path.is_file():
        raise FileNotFoundError(f"Missing enron edge list: {edge_path}")
    frame = pd.read_csv(edge_path)
    required = {"u", "i", "ts"}
    if not required.issubset(frame.columns):
        raise ValueError("enron edge list must contain u, i, ts columns.")

    source = frame["u"].to_numpy(dtype=np.int64)
    destination_raw = frame["i"].to_numpy(dtype=np.int64)
    timestamps = frame["ts"].to_numpy(dtype=np.int64)
    destination_offset = int(source.max()) + 1
    destination = destination_raw + destination_offset
    num_nodes = int(np.maximum(source.max(), destination.max())) + 1

    val_negatives = _load_negative_dict(dataset_dir / "tgbl-enron_val_ns.pkl")
    test_negatives = _load_negative_dict(dataset_dir / "tgbl-enron_test_ns.pkl")
    val_keys = set(val_negatives)
    test_keys = set(test_negatives)
    val_mask = np.fromiter(
        (
            (int(u), int(i) + destination_offset, int(t)) in val_keys
            for u, i, t in zip(source, destination_raw, timestamps)
        ),
        dtype=bool,
        count=len(source),
    )
    test_mask = np.fromiter(
        (
            (int(u), int(i) + destination_offset, int(t)) in test_keys
            for u, i, t in zip(source, destination_raw, timestamps)
        ),
        dtype=bool,
        count=len(source),
    )
    train_mask = ~(val_mask | test_mask)

    data = TemporalData(
        src=torch.from_numpy(source),
        dst=torch.from_numpy(destination),
        t=torch.from_numpy(timestamps),
        msg=torch.zeros(len(source), 1, dtype=torch.float32),
    )
    return build_temporal_graph_bundle(
        dataset_name="tgbl-enron",
        data=data,
        node_features=torch.zeros(num_nodes, feature_dim),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
        feature_dim=feature_dim,
        official_evaluation=max_events <= 0,
        tgb_dataset=_EnronBackend(val_negatives, test_negatives),
        max_events=max_events,
        structure_feature_set=structure_feature_set,
    )
