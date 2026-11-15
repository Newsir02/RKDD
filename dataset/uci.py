from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import TemporalData

from .temporal import TemporalGraphBundle, build_temporal_graph_bundle


class _UCINegativeSampler:
    """Fixed random negatives for the chronological UCI benchmark.

    UCI has no TGB-style official negative-sampling file.  We therefore use
    the standard chronological 70/15/15 split and pre-sample a fixed set of
    destination negatives for every validation and test event.
    """

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
    ) -> List[np.ndarray]:
        table = self.val_negatives if split_mode == "val" else self.test_negatives
        result = []
        for source, destination, timestamp in zip(src, dst, timestamps):
            key = (int(source), int(destination), int(timestamp))
            if key not in table:
                raise KeyError(f"Missing UCI negative samples for event {key}.")
            result.append(table[key].copy())
        return result


class _UCIBackend:
    eval_metric = "mrr"

    def __init__(
        self,
        val_negatives: Dict[tuple[int, int, int], np.ndarray],
        test_negatives: Dict[tuple[int, int, int], np.ndarray],
    ) -> None:
        self.negative_sampler = _UCINegativeSampler(
            val_negatives,
            test_negatives,
        )

    def load_val_ns(self) -> None:
        return None

    def load_test_ns(self) -> None:
        return None


def _sample_negatives(
    sources: np.ndarray,
    destinations: np.ndarray,
    timestamps: np.ndarray,
    num_nodes: int,
    count: int,
    seed: int,
) -> Dict[tuple[int, int, int], np.ndarray]:
    if count <= 0:
        raise ValueError("num_negatives must be positive.")
    generator = np.random.default_rng(seed)
    result: Dict[tuple[int, int, int], np.ndarray] = {}
    for source, positive, timestamp in zip(
        sources.tolist(),
        destinations.tolist(),
        timestamps.tolist(),
    ):
        chosen: List[int] = []
        while len(chosen) < count:
            candidate = int(generator.integers(0, num_nodes))
            if candidate == source or candidate == positive:
                continue
            chosen.append(candidate)
        result[(int(source), int(positive), int(timestamp))] = np.asarray(
            chosen,
            dtype=np.int64,
        )
    return result


def _chronological_masks(
    timestamps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(timestamps)
    train_end = int(count * 0.70)
    val_end = int(count * 0.85)
    train_mask = np.zeros(count, dtype=bool)
    val_mask = np.zeros(count, dtype=bool)
    test_mask = np.zeros(count, dtype=bool)
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True
    return train_mask, val_mask, test_mask


def build_uci(
    *,
    name: str = "uci",
    root: str = "datasets",
    feature_dim: int = 16,
    edge_feature_dim: int = 16,
    num_candidates: int = 16,
    history_size: int = 24,
    seed: int = 42,
    max_events: int = 0,
    download: bool = True,
    structure_feature_set: str = "basic",
    num_negatives: int = 20,
) -> TemporalGraphBundle:
    del edge_feature_dim, num_candidates, history_size, download
    if name != "uci":
        raise ValueError("build_uci only supports 'uci'.")

    dataset_dir = Path(root) / "uci"
    edge_path = dataset_dir / "ml_uci.csv"
    edge_feature_path = dataset_dir / "ml_uci.npy"
    node_feature_path = dataset_dir / "ml_uci_node.npy"
    for path in (edge_path, edge_feature_path, node_feature_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing UCI file: {path}")

    frame = pd.read_csv(edge_path)
    required = {"u", "i", "ts"}
    if not required.issubset(frame.columns):
        raise ValueError("UCI edge list must contain u, i, ts columns.")

    source = frame["u"].to_numpy(dtype=np.int64)
    destination = frame["i"].to_numpy(dtype=np.int64)
    timestamps = frame["ts"].to_numpy(dtype=np.int64)
    order = np.argsort(timestamps, kind="stable")
    source = source[order]
    destination = destination[order]
    timestamps = timestamps[order]

    edge_features = np.load(edge_feature_path).astype(np.float32)
    node_features = np.load(node_feature_path).astype(np.float32)
    if edge_features.shape[0] < len(source):
        raise ValueError("UCI edge feature rows do not cover the edge list.")
    edge_features = edge_features[: len(source)][order]
    num_nodes = max(int(source.max()), int(destination.max()), node_features.shape[0] - 1) + 1
    if node_features.shape[0] < num_nodes:
        raise ValueError("UCI node features do not cover all node ids.")

    train_mask, val_mask, test_mask = _chronological_masks(timestamps)
    val_negatives = _sample_negatives(
        source[val_mask],
        destination[val_mask],
        timestamps[val_mask],
        num_nodes,
        num_negatives,
        seed + 1,
    )
    test_negatives = _sample_negatives(
        source[test_mask],
        destination[test_mask],
        timestamps[test_mask],
        num_nodes,
        num_negatives,
        seed + 2,
    )

    try:
        import tgb.utils.info as info

        info.DATA_EVAL_METRIC_DICT.setdefault("uci", "mrr")
    except ImportError:
        pass

    data = TemporalData(
        src=torch.from_numpy(source),
        dst=torch.from_numpy(destination),
        t=torch.from_numpy(timestamps),
        msg=torch.from_numpy(edge_features),
    )
    return build_temporal_graph_bundle(
        dataset_name="uci",
        data=data,
        node_features=torch.from_numpy(node_features),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
        feature_dim=feature_dim,
        official_evaluation=max_events <= 0,
        tgb_dataset=_UCIBackend(val_negatives, test_negatives),
        max_events=max_events,
        structure_feature_set=structure_feature_set,
    )
