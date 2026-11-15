from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dataset import (
    CandidateBatchLoader,
    TemporalGraphBundle,
    build_synthetic_temporal_graph,
    build_tgb_enron,
    build_tgb_temporal_graph,
    build_uci,
)
from loss import (
    DistillationObjective,
    masked_relation_mean,
    relation_alignment_loss,
    rkd_angle_loss,
    rkd_distance_loss,
)
from loss.temporal_delta import TemporalDeltaDistillation
from util.temporal_delta_training import train_temporal_delta_epoch
from model import (
    EnsembleStudent,
    EnsembleTeacher,
    GraphMixerTeacher,
    LightGNNStudent,
    PlainGNNStudent,
    PyGTGNTeacher,
    TGATTeacher,
)
from util import (
    AverageMeter,
    RankingMetricAccumulator,
    benchmark_latency,
    evaluate_tgb,
    load_model_checkpoint,
    model_parameter_summary,
    move_to_device,
    save_model_checkpoint,
    seed_everything,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TGB temporal link distillation with a PyG TGN teacher and "
            "a lightweight temporal GNN student."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("teacher", "distill", "student", "all"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dataset", type=str, default="tgbl-wiki")
    parser.add_argument("--data-root", type=str, default="datasets")
    parser.add_argument("--max-real-events", type=int, default=0)
    parser.add_argument(
        "--no-tgb-download",
        action="store_false",
        dest="tgb_download",
    )
    parser.set_defaults(tgb_download=True)

    parser.add_argument("--num-nodes", type=int, default=256)
    parser.add_argument("--num-events", type=int, default=3000)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--edge-feature-dim", type=int, default=16)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--history-size", type=int, default=24)
    parser.add_argument("--temporal-neighbors", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--teacher-arch",
        choices=("tgn", "graphmixer", "tgat"),
        default="tgn",
        help="教师架构：tgn=PyG TGN；graphmixer=无 memory 的邻居混合；tgat=时序图注意力。",
    )
    parser.add_argument("--teacher-hidden", type=int, default=64)
    parser.add_argument("--teacher-layers", type=int, default=1)
    parser.add_argument(
        "--teacher-node-features",
        action="store_true",
        help=(
            "把静态节点特征投影后注入 TGN 教师查询表示；"
            "默认关闭以保持旧检查点与历史结果完全一致。"
        ),
    )
    parser.add_argument("--student-hidden", type=int, default=32)
    parser.add_argument(
        "--student-blocks",
        "--student-layers",
        dest="student_blocks",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--student-arch",
        choices=("gnn",),
        default="gnn",
    )
    parser.add_argument("--student-neighbors", type=int, default=8)
    parser.add_argument(
        "--student-aggregation",
        choices=("mean", "kernel"),
        default="mean",
        help=(
            "GNN 学生的邻居聚合方式：mean=均匀均值（旧行为）；"
            "kernel=内容/时间感知的单头注意力聚合，可被时序关系影响力蒸馏监督。"
        ),
    )
    parser.add_argument(
        "--student-kernel-hidden",
        type=int,
        default=32,
        help="学生时序关系注意力模块的隐藏维度。",
    )
    parser.add_argument(
        "--student-neighbor-store",
        choices=("exact", "pyg"),
        default="exact",
        help=(
            "邻居历史存储：exact=保留最近 K 条（修复 PyG 环形缓冲覆盖问题）；"
            "pyg=PyG LastNeighborLoader，仅用于复现旧结果。"
        ),
    )
    parser.add_argument(
        "--structure-features",
        action="store_true",
        default=None,
        help=(
            "把离线统计的结构/时序节点特征拼接到学生输入，"
            "使图无关学生也具备结构/近期活跃度感知。"
        ),
    )
    parser.add_argument(
        "--structure-feature-set",
        choices=("basic", "rich", "nbr", "nbr2", "nbr3", "rich_nbr"),
        default="basic",
        help=(
            "结构/时序特征集合：basic=5 维手工特征；rich=12 维手工特征；"
            "nbr/nbr2/nbr3=basic+1/2/3 跳邻居特征均值；rich_nbr=rich+1 跳。"
        ),
    )
    parser.add_argument("--time-dim", type=int, default=16)
    parser.add_argument("--relation-dim", type=int, default=64)
    parser.add_argument("--relation-slots", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--student-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--teacher-lr", type=float, default=1.0e-3)
    parser.add_argument("--student-lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help=(
            "学生权重的 EMA 衰减系数：>0 启用（如 0.999），用于降低 seed 方差。"
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument("--logit-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--diversity-weight", type=float, default=0.0)
    # 默认 None 的权重由 apply_method_configuration 按 --method 填充。
    parser.add_argument("--anchor-weight", type=float, default=None)
    parser.add_argument("--teacher-relation-weight", type=float, default=None)
    parser.add_argument(
        "--kernel-weight",
        type=float,
        default=None,
        help=(
            "时序关系影响力蒸馏权重：在共享历史事件支撑上对齐教师注意力质量"
            "（教师的离散条件强度 / Hawkes 式影响核）。"
        ),
    )
    parser.add_argument(
        "--kernel-divergence",
        choices=("kl", "js", "mse"),
        default="kl",
        help="影响力分布匹配的散度；mse 为 attention-transfer 式对照。",
    )
    parser.add_argument(
        "--kernel-teacher-mode",
        choices=("teacher", "uniform", "recency", "shuffled"),
        default="teacher",
        help=(
            "TRID 的教师权重对照：teacher=使用教师输出；uniform=均匀权重；"
            "recency=按年龄的指数衰减；shuffled=行内随机打乱教师权重。"
        ),
    )
    parser.add_argument(
        "--dkd-weight",
        type=float,
        default=0.0,
        help="DKD（CVPR 2022）解耦 logit 蒸馏权重：分离目标类与非目标类知识。",
    )
    parser.add_argument(
        "--dkd-tckd-weight",
        type=float,
        default=1.0,
        help="DKD 中目标类知识（TCKD）权重。",
    )
    parser.add_argument(
        "--dkd-nckd-weight",
        type=float,
        default=1.0,
        help="DKD 中非目标类知识（NCKD）权重。",
    )
    parser.add_argument(
        "--rkd-weight",
        type=float,
        default=0.0,
        help="RKD 候选关系蒸馏权重：对齐候选嵌入的距离/角度结构。",
    )
    parser.add_argument(
        "--rkd-mode",
        choices=("distance", "angle", "both"),
        default="both",
        help="RKD 关系类型：距离、角度或两者平均。",
    )
    parser.add_argument(
        "--confidence-weight",
        type=float,
        default=0.0,
        help="教师置信度门控指数：>0 时按 teacher top-1/top-2 margin 加权 logit/rank KD。",
    )
    parser.add_argument(
        "--method",
        choices=("baseline", "trid", "dkd", "trd"),
        default="baseline",
    )
    parser.add_argument("--trd-weight", type=float, default=1.0,
                        help="TRD lambda_2; zero gives output KD on the same plain GNN.")
    parser.add_argument("--trd-mode", choices=TemporalDeltaDistillation.MODES, default="temporal")
    parser.add_argument("--student-seed", type=int, default=None)
    parser.add_argument("--logit-temperature", type=float, default=2.0)
    parser.add_argument("--rank-temperature", type=float, default=1.0)

    parser.add_argument("--teacher-checkpoint", type=str, default="")
    parser.add_argument(
        "--teacher-checkpoints",
        type=str,
        default="",
        help=(
            "逗号分隔的多个教师 checkpoint，用于多教师 logit 集成蒸馏；"
            "设置后优先于 --teacher-checkpoint。"
        ),
    )
    parser.add_argument(
        "--mosa-teacher-checkpoints",
        type=str,
        default="",
        help=(
            "逗号分隔的多个学生 checkpoint；把它们的平均预测作为教师，"
            "蒸馏回一个单学生（MOSA-S）。"
        ),
    )
    parser.add_argument(
        "--mosa-rkd-weight",
        type=float,
        default=0.0,
        help=(
            "MOSA-S 候选几何对齐权重：新学生同时匹配每个教师学生的"
            "候选距离/角度关系。"
        ),
    )
    parser.add_argument(
        "--teacher-query-neighbors",
        type=int,
        default=0,
        help=(
            "只改变教师最近邻队列长度（0=用 checkpoint 配置）；"
            "用于让教师的打分支撑与学生一致（support-matched KD）。"
        ),
    )
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--latency-batches", type=int, default=20)
    parser.add_argument(
        "--eval-max-events",
        type=int,
        default=0,
        help=(
            "开发期加速：官方评测时 val 只用前 N 个事件（0=全量）。"
            "test 始终全量，不影响最终报告。"
        ),
    )
    args = parser.parse_args(argv)
    args.student_layers = args.student_blocks
    return args


def validate_args(args: argparse.Namespace) -> None:
    ensemble_paths = [
        item.strip()
        for item in str(getattr(args, "teacher_checkpoints", "")).split(",")
        if item.strip()
    ]
    mosa_paths = [
        item.strip()
        for item in str(
            getattr(args, "mosa_teacher_checkpoints", "")
        ).split(",")
        if item.strip()
    ]
    if mosa_paths and (args.teacher_checkpoint or ensemble_paths):
        raise ValueError(
            "--mosa-teacher-checkpoints cannot be combined with "
            "--teacher-checkpoint/--teacher-checkpoints."
        )
    if (
        args.stage == "distill"
        and not args.teacher_checkpoint
        and not ensemble_paths
        and not mosa_paths
    ):
        raise ValueError(
            "--stage distill requires --teacher-checkpoint, "
            "--teacher-checkpoints, or --mosa-teacher-checkpoints."
        )
    if (
        args.stage == "distill"
        and not ensemble_paths
        and not mosa_paths
        and not Path(args.teacher_checkpoint).is_file()
    ):
        raise FileNotFoundError(
            f"Teacher checkpoint does not exist: {args.teacher_checkpoint}"
        )
    for path in ensemble_paths + mosa_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Teacher checkpoint does not exist: {path}")
    if float(getattr(args, "mosa_rkd_weight", 0.0)) < 0.0:
        raise ValueError("mosa-rkd-weight must be non-negative.")
    if float(getattr(args, "mosa_rkd_weight", 0.0)) > 0.0 and not mosa_paths:
        raise ValueError(
            "--mosa-rkd-weight requires --mosa-teacher-checkpoints."
        )
    if mosa_paths:
        unsupported = [
            name
            for name in (
                "anchor_weight",
                "kernel_weight",
                "rkd_weight",
            )
            if float(getattr(args, name, 0.0) or 0.0) != 0.0
        ]
        if unsupported:
            raise ValueError(
                "MOSA-S supports logit/rank objectives only; disable: "
                + ", ".join(unsupported)
            )
    positive_names = {
        "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs,
        "feature_dim": args.feature_dim,
        "edge_feature_dim": args.edge_feature_dim,
        "num_candidates": args.num_candidates,
        "temporal_neighbors": args.temporal_neighbors,
        "batch_size": args.batch_size,
        "teacher_layers": args.teacher_layers,
        "student_blocks": args.student_blocks,
        "latency_batches": args.latency_batches,
    }
    invalid = [name for name, value in positive_names.items() if value <= 0]
    if invalid:
        raise ValueError("These arguments must be positive: " + ", ".join(invalid))
    if args.num_candidates < 2:
        raise ValueError("num-candidates must be at least 2.")
    if args.patience < 0:
        raise ValueError("patience must be non-negative.")
    if args.eval_max_events < 0:
        raise ValueError("eval-max-events must be non-negative.")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema-decay must be in [0, 1).")
    if args.student_neighbors <= 0:
        raise ValueError("student-neighbors must be positive.")
    if int(getattr(args, "student_kernel_hidden", 1)) <= 0:
        raise ValueError("student-kernel-hidden must be positive.")
    kernel_weight = float(getattr(args, "kernel_weight", 0.0) or 0.0)
    if kernel_weight < 0.0:
        raise ValueError("kernel-weight must be non-negative.")
    if kernel_weight > 0.0 and args.student_arch != "gnn":
        raise ValueError("kernel distillation requires --student-arch gnn.")
    if kernel_weight > 0.0 and args.student_aggregation != "kernel":
        raise ValueError(
            "kernel distillation requires --student-aggregation kernel."
        )
    if float(getattr(args, "dkd_weight", 0.0)) < 0.0:
        raise ValueError("dkd-weight must be non-negative.")
    if float(getattr(args, "dkd_tckd_weight", 1.0)) < 0.0:
        raise ValueError("dkd-tckd-weight must be non-negative.")
    if float(getattr(args, "dkd_nckd_weight", 1.0)) < 0.0:
        raise ValueError("dkd-nckd-weight must be non-negative.")
    if float(getattr(args, "rkd_weight", 0.0)) < 0.0:
        raise ValueError("rkd-weight must be non-negative.")
    if float(getattr(args, "confidence_weight", 0.0)) < 0.0:
        raise ValueError("confidence-weight must be non-negative.")
    if args.teacher_hidden % 2 != 0:
        raise ValueError("teacher-hidden must be even for TransformerConv heads.")


def apply_quick_configuration(args: argparse.Namespace) -> None:
    """把实验压缩为 CPU 可运行的非官方 smoke 配置。"""

    args.num_nodes = min(args.num_nodes, 64)
    args.num_events = min(args.num_events, 120)
    args.num_candidates = min(args.num_candidates, 8)
    args.history_size = min(args.history_size, 8)
    args.temporal_neighbors = min(args.temporal_neighbors, 8)
    args.batch_size = min(args.batch_size, 24)
    args.teacher_hidden = min(args.teacher_hidden, 32)
    args.student_hidden = min(args.student_hidden, 32)
    args.relation_dim = min(args.relation_dim, 32)
    args.relation_slots = min(args.relation_slots, 4)
    args.student_neighbors = min(args.student_neighbors, 8)
    args.teacher_epochs = 1
    args.student_epochs = 1
    args.latency_batches = 1
    args.student_layers = args.student_blocks
    if args.dataset != "synthetic":
        args.max_real_events = min(args.max_real_events or 120, 120)


def apply_method_configuration(args: argparse.Namespace) -> None:
    """把 --method 映射为新增权重；显式传入的参数优先。"""

    if args.method == "trd":
        if args.teacher_checkpoints or args.mosa_teacher_checkpoints:
            raise ValueError("TRD requires a single temporal teacher.")
        if args.structure_features or args.student_aggregation != "mean":
            raise ValueError("TRD uses a plain mean GNN without offline temporal features.")
        # No legacy objective or auxiliary teacher loss can enter the new path.
        for name in ("rank_weight", "diversity_weight", "anchor_weight",
                     "teacher_relation_weight", "kernel_weight", "rkd_weight",
                     "confidence_weight", "dkd_weight", "mosa_rkd_weight"):
            setattr(args, name, 0.0)
        args.task_weight = 1.0
        args.structure_features = False
        args.student_neighbor_store = "exact"
        args.dropout = 0.0
        # AdamW regularization is disabled in the strict three-term recipe.
        args.weight_decay = 0.0
        TemporalDeltaDistillation(args.logit_weight, args.trd_weight,
                                  args.logit_temperature, args.trd_mode)
    elif args.method == "dkd":
        args.logit_weight = 0.0
        args.rank_weight = 0.0
        args.anchor_weight = 0.0
        args.teacher_relation_weight = 0.0
        args.kernel_weight = 0.0
        args.diversity_weight = 0.0
        if float(getattr(args, "dkd_weight", 0.0)) == 0.0:
            args.dkd_weight = 1.0
    elif args.method == "trid":
        # Temporal Relational Influence Distillation：只新增影响力匹配项，
        # 其余分量沿用显式传入的权重。
        if args.anchor_weight is None:
            args.anchor_weight = 0.0
        if args.teacher_relation_weight is None:
            args.teacher_relation_weight = 1.0
        if args.kernel_weight is None:
            args.kernel_weight = 1.0
    else:
        if args.anchor_weight is None:
            args.anchor_weight = 0.0
        if args.teacher_relation_weight is None:
            args.teacher_relation_weight = 0.0
        if args.kernel_weight is None:
            args.kernel_weight = 0.0
    args.structure_features = bool(args.structure_features)


def resolve_student_seed(args: argparse.Namespace) -> int:
    """学生初始化/训练随机种子；未指定时回退到 --seed 以保持旧命令兼容。"""

    if args.student_seed is None:
        args.student_seed = args.seed
    return int(args.student_seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_loaders(
    args: argparse.Namespace,
) -> tuple[
    TemporalGraphBundle,
    CandidateBatchLoader,
    CandidateBatchLoader,
    CandidateBatchLoader,
]:
    if args.dataset == "synthetic":
        bundle = build_synthetic_temporal_graph(
            num_nodes=args.num_nodes,
            num_events=args.num_events,
            feature_dim=args.feature_dim,
            edge_feature_dim=args.edge_feature_dim,
            num_candidates=args.num_candidates,
            history_size=args.history_size,
            seed=args.seed,
            structure_feature_set=args.structure_feature_set,
        )
    elif args.dataset == "uci":
        bundle = build_uci(
            name=args.dataset,
            root=args.data_root,
            feature_dim=args.feature_dim,
            edge_feature_dim=args.edge_feature_dim,
            num_candidates=args.num_candidates,
            history_size=args.history_size,
            seed=args.seed,
            max_events=args.max_real_events,
            download=args.tgb_download,
            structure_feature_set=args.structure_feature_set,
        )
    elif args.dataset == "tgbl-enron":
        bundle = build_tgb_enron(
            name=args.dataset,
            root=args.data_root,
            feature_dim=args.feature_dim,
            edge_feature_dim=args.edge_feature_dim,
            num_candidates=args.num_candidates,
            history_size=args.history_size,
            seed=args.seed,
            max_events=args.max_real_events,
            download=args.tgb_download,
            structure_feature_set=args.structure_feature_set,
        )
    else:
        bundle = build_tgb_temporal_graph(
            name=args.dataset,
            root=args.data_root,
            feature_dim=args.feature_dim,
            edge_feature_dim=args.edge_feature_dim,
            num_candidates=args.num_candidates,
            history_size=args.history_size,
            seed=args.seed,
            max_events=args.max_real_events,
            download=args.tgb_download,
            structure_feature_set=args.structure_feature_set,
        )
    common = {
        "bundle": bundle,
        "batch_size": args.batch_size,
        "num_candidates": args.num_candidates,
        "num_workers": args.num_workers,
    }
    train_loader = CandidateBatchLoader(
        split="train",
        seed=args.seed,
        **common,
    )
    validation_loader = CandidateBatchLoader(
        split="val",
        seed=args.seed + 1,
        **common,
    )
    test_loader = CandidateBatchLoader(
        split="test",
        seed=args.seed + 2,
        **common,
    )
    return bundle, train_loader, validation_loader, test_loader


def _update_signature_with_tensor(hasher: object, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(str(tuple(value.shape)).encode("ascii"))
    hasher.update(value.numpy().tobytes())


def compute_data_signature(
    args: argparse.Namespace,
    bundle: TemporalGraphBundle,
) -> str:
    """对模型实际使用的官方事件流、特征和切分生成稳定指纹。"""

    hasher = hashlib.sha256()
    metadata = {
        # Historical data-format identifier: changing it invalidates existing checkpoints.
        "format": "cutfgwd-pyg-temporal-v2",
        "dataset": args.dataset,
        "seed": args.seed,
        "feature_dim": args.feature_dim,
        "num_candidates": args.num_candidates,
        "max_real_events": args.max_real_events,
        "official_evaluation": bundle.official_evaluation,
    }
    hasher.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    _update_signature_with_tensor(hasher, bundle.node_features)
    for name in ("src", "dst", "t", "msg", "event_index", "timestamp"):
        hasher.update(name.encode("ascii"))
        _update_signature_with_tensor(hasher, getattr(bundle.data, name))
    for name, split_data in (
        ("train", bundle.train_data),
        ("val", bundle.val_data),
        ("test", bundle.test_data),
    ):
        hasher.update(name.encode("ascii"))
        _update_signature_with_tensor(hasher, split_data.event_index)
    return hasher.hexdigest()


def load_student_from_checkpoint(
    checkpoint_path: str | Path,
    bundle: TemporalGraphBundle,
    device: torch.device,
    *,
    data_signature: str | None = None,
) -> LightGNNStudent | PlainGNNStudent:
    """Load a student for independent evaluation or ensembling."""

    payload = load_model_checkpoint(checkpoint_path)
    config = payload["model_config"]
    saved_signature = config.get("data_signature")
    if data_signature is not None and saved_signature is not None and saved_signature != data_signature:
        raise ValueError(f"Student checkpoint data_signature mismatch: {checkpoint_path}")
    if payload["model_name"] == "PlainGNNStudent":
        student = PlainGNNStudent(bundle.node_features, int(config["hidden_dim"]),
                                  int(config["student_neighbors"]), int(config["num_blocks"]))
        student.load_state_dict(payload["model"], strict=True)
        return student.to(device).eval()
    if payload["model_name"] != "LightGNNStudent":
        raise ValueError(
            "Expected a LightGNNStudent or PlainGNNStudent checkpoint, got "
            f"{payload['model_name']!r}."
        )
    features = bundle.node_features
    expected_dim = int(config.get("node_feature_dim", features.size(1)))
    if features.size(1) != expected_dim:
        if (
            bundle.structure_features is not None
            and features.size(1) + bundle.structure_features.size(1) == expected_dim
        ):
            features = torch.cat(
                [features, bundle.structure_features], dim=-1
            )
        else:
            raise ValueError(
                "Student node feature dimension does not match checkpoint: "
                f"{features.size(1)} vs {expected_dim}."
            )
    time_span = float(
        (
            bundle.data.t[-1].double() - bundle.data.t[0].double()
        ).clamp_min(1.0)
    )
    student = LightGNNStudent(
        node_features=features,
        hidden_dim=int(config["hidden_dim"]),
        time_dim=int(config["time_dim"]),
        relation_dim=int(config["relation_dim"]),
        relation_slots=int(config["relation_slots"]),
        dropout=float(config["dropout"]),
        num_blocks=int(config["num_blocks"]),
        neighbors=int(config["student_neighbors"]),
        aggregation=str(config["student_aggregation"]),
        kernel_hidden=int(config.get("student_kernel_hidden", 32)),
        event_times=bundle.data.t,
        time_span=time_span,
        neighbor_store=str(config.get("student_neighbor_store", "exact")),
    )
    state = dict(payload["model"])
    # Ignore only known retired training heads; validate all active weights strictly.
    state.pop("embed_projector.weight", None)
    state.pop("embed_projector.bias", None)
    state.pop("relation_mass_head.weight", None)
    state.pop("relation_mass_head.bias", None)
    student.load_state_dict(state, strict=True)
    student.reset_state()
    return student.to(device).eval()


def build_teacher(
    bundle: TemporalGraphBundle,
    args: argparse.Namespace,
) -> nn.Module:
    node_features = (
        bundle.node_features
        if bool(getattr(args, "teacher_node_features", False))
        else None
    )
    teacher_arch = getattr(args, "teacher_arch", "tgn")
    time_span = None
    if getattr(args, "method", "baseline") == "trd" and teacher_arch in ("graphmixer", "tgat"):
        # Fit the time scale to the training prefix for newly trained TRD teachers.
        time_span = max(1.0, float(bundle.train_data.t[-1] - bundle.train_data.t[0]))
    if teacher_arch == "graphmixer":
        return GraphMixerTeacher(
            event_times=bundle.data.t,
            event_messages=bundle.data.msg,
            num_nodes=int(bundle.node_features.size(0)),
            hidden_dim=args.teacher_hidden,
            time_dim=args.time_dim,
            relation_dim=args.relation_dim,
            temporal_neighbors=args.temporal_neighbors,
            dropout=args.dropout,
            node_features=node_features,
            time_span=time_span,
        )
    if teacher_arch == "tgat":
        return TGATTeacher(
            event_times=bundle.data.t,
            event_messages=bundle.data.msg,
            num_nodes=int(bundle.node_features.size(0)),
            hidden_dim=args.teacher_hidden,
            time_dim=args.time_dim,
            relation_dim=args.relation_dim,
            temporal_neighbors=args.temporal_neighbors,
            dropout=args.dropout,
            node_features=node_features,
            time_span=time_span,
        )
    return PyGTGNTeacher(
        event_times=bundle.data.t,
        event_messages=bundle.data.msg,
        num_nodes=int(bundle.node_features.size(0)),
        hidden_dim=args.teacher_hidden,
        time_dim=args.time_dim,
        relation_dim=args.relation_dim,
        temporal_neighbors=args.temporal_neighbors,
        dropout=args.dropout,
        teacher_layers=args.teacher_layers,
        node_features=node_features,
    )


def build_student(
    node_features: Tensor,
    args: argparse.Namespace,
    structure_features: Optional[Tensor] = None,
    bundle: Optional[TemporalGraphBundle] = None,
) -> nn.Module:
    if args.method == "trd":
        return PlainGNNStudent(node_features, args.student_hidden,
                               args.student_neighbors, args.student_blocks)
    student_features = node_features
    if (
        bool(getattr(args, "structure_features", False))
        and structure_features is not None
    ):
        # 拼接离线结构/时序特征；这些特征只从训练集统计，推理时无需访问图。
        student_features = torch.cat(
            [
                node_features.to(torch.float32),
                structure_features.to(torch.float32),
            ],
            dim=-1,
        )
    event_times = None
    time_span = None
    if bundle is not None:
        event_times = bundle.data.t
        time_span = float(
            (bundle.data.t[-1].double() - bundle.data.t[0].double()).clamp_min(1.0)
        )
    return LightGNNStudent(
        node_features=student_features,
        hidden_dim=args.student_hidden,
        time_dim=args.time_dim,
        relation_dim=args.relation_dim,
        relation_slots=args.relation_slots,
        dropout=args.dropout,
        num_blocks=args.student_blocks,
        neighbors=args.student_neighbors,
        aggregation=args.student_aggregation,
        kernel_hidden=args.student_kernel_hidden,
        event_times=event_times,
        time_span=time_span,
        neighbor_store=args.student_neighbor_store,
    )


def load_teacher_from_checkpoint(
    checkpoint_path: str | Path,
    bundle: TemporalGraphBundle,
    data_signature: str,
    device: torch.device,
    temporal_neighbors: Optional[int] = None,
) -> nn.Module:
    """加载教师检查点；支持 TGN 与 GraphMixer 教师。"""

    payload = load_model_checkpoint(checkpoint_path)
    config = payload["model_config"]
    model_name = str(payload["model_name"])

    expected_config = {
        "num_nodes": int(bundle.node_features.size(0)),
        "num_events": int(bundle.data.num_events),
        "msg_dim": int(bundle.data.msg.size(1)),
        "data_signature": data_signature,
    }
    for name, expected_value in expected_config.items():
        actual_value = config.get(name)
        if actual_value != expected_value:
            raise ValueError(
                f"Checkpoint config mismatch for {name!r}: "
                f"got {actual_value!r}, expected {expected_value!r}."
            )
    required = {
        "hidden_dim",
        "time_dim",
        "relation_dim",
        "temporal_neighbors",
        "dropout",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(
            "Teacher checkpoint config is missing: " + ", ".join(missing)
        )
    resolved_neighbors = int(
        temporal_neighbors
        if temporal_neighbors is not None
        else config["temporal_neighbors"]
    )
    node_features = (
        bundle.node_features
        if bool(config.get("teacher_node_features", False))
        else None
    )
    if model_name == "GraphMixerTeacher":
        teacher: nn.Module = GraphMixerTeacher(
            event_times=bundle.data.t,
            event_messages=bundle.data.msg,
            num_nodes=int(config["num_nodes"]),
            hidden_dim=int(config["hidden_dim"]),
            time_dim=int(config["time_dim"]),
            relation_dim=int(config["relation_dim"]),
            temporal_neighbors=resolved_neighbors,
            dropout=float(config["dropout"]),
            node_features=node_features,
            time_span=config.get("time_span"),
        )
    elif model_name == "TGATTeacher":
        teacher = TGATTeacher(
            event_times=bundle.data.t,
            event_messages=bundle.data.msg,
            num_nodes=int(config["num_nodes"]),
            hidden_dim=int(config["hidden_dim"]),
            time_dim=int(config["time_dim"]),
            relation_dim=int(config["relation_dim"]),
            temporal_neighbors=resolved_neighbors,
            dropout=float(config["dropout"]),
            heads=int(config.get("teacher_heads", 2)),
            node_features=node_features,
            time_span=config.get("time_span"),
        )
    elif model_name == "PyGTGNTeacher":
        teacher = PyGTGNTeacher(
            event_times=bundle.data.t,
            event_messages=bundle.data.msg,
            num_nodes=int(config["num_nodes"]),
            hidden_dim=int(config["hidden_dim"]),
            time_dim=int(config["time_dim"]),
            relation_dim=int(config["relation_dim"]),
            temporal_neighbors=resolved_neighbors,
            dropout=float(config["dropout"]),
            teacher_layers=int(config.get("teacher_layers", 1)),
            node_features=node_features,
        )
    else:
        raise ValueError(
            "Unsupported teacher checkpoint model: "
            f"{model_name!r}; expected GraphMixerTeacher, TGATTeacher, or PyGTGNTeacher."
        )

    try:
        teacher.load_state_dict(payload["model"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Teacher checkpoint parameters are incompatible: {exc}"
        ) from exc
    reset_state = getattr(teacher, "reset_state", None)
    if callable(reset_state):
        reset_state()
    return teacher.to(device)


@torch.no_grad()
def evaluate_smoke(
    model: nn.Module,
    loader: CandidateBatchLoader,
    device: torch.device,
) -> Dict[str, object]:
    """在固定随机候选上运行非官方 smoke 指标。"""

    model.eval()
    loss_meter = AverageMeter()
    metrics = RankingMetricAccumulator()
    score_candidates = getattr(model, "score_candidates", None)
    update_state = getattr(model, "update_state", None)
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        if callable(score_candidates):
            output = score_candidates(batch, return_relations=False)
        else:
            output = model(batch, return_relations=False)
        loss = F.cross_entropy(output["logits"], batch["target"])
        count = int(batch["target"].numel())
        loss_meter.update(float(loss), count)
        metrics.update(output["logits"], batch["target"])
        if callable(update_state):
            update_state(batch)
    result: Dict[str, object] = metrics.compute()
    result.update(
        {
            "metric": "mrr",
            "loss": loss_meter.average,
            "official": False,
        }
    )
    return result


def evaluate_split(
    model: nn.Module,
    bundle: TemporalGraphBundle,
    loader: CandidateBatchLoader,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    if bundle.official_evaluation:
        if split not in {"val", "test"}:
            raise ValueError("Official evaluation supports val or test only.")
        # --eval-max-events 只用于加速开发期的 val 选择；test 始终全量。
        max_events = (
            int(getattr(args, "eval_max_events", 0)) if split == "val" else 0
        )
        return evaluate_tgb(
            model,
            bundle,
            split,
            args.batch_size,
            device,
            max_events=max_events,
        )
    return evaluate_smoke(model, loader, device)


@torch.no_grad()
def replay_teacher(
    teacher: nn.Module,
    loaders: Iterable[CandidateBatchLoader],
    device: torch.device,
) -> None:
    """只回放真实正边，重建教师最近邻状态。"""

    teacher.eval()
    for loader in loaders:
        for raw_batch in loader:
            teacher.update_state(move_to_device(raw_batch, device))


def evaluate_teacher_from_history(
    teacher: nn.Module,
    bundle: TemporalGraphBundle,
    history_loaders: Iterable[CandidateBatchLoader],
    target_loader: CandidateBatchLoader,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    teacher.eval()
    teacher.reset_state()
    replay_teacher(teacher, history_loaders, device)
    return evaluate_split(
        teacher,
        bundle,
        target_loader,
        split,
        args,
        device,
    )


def train_teacher_epoch(
    teacher: nn.Module,
    loader: CandidateBatchLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    relation_aux_weight: float = 0.0,
) -> float:
    teacher.train()
    teacher.reset_state()
    # 关系对齐头始终存在；仅当辅助权重非零时才请求关系输出并计算辅助损失。
    use_relations = relation_aux_weight > 0.0
    meter = AverageMeter()
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = teacher.score_candidates(batch, return_relations=use_relations)
        loss = F.cross_entropy(output["logits"], batch["target"])
        if use_relations:
            # 自监督：池化关系表示应能重建教师打分所用的 source embedding。
            pooled = masked_relation_mean(
                output["relation_tokens"],
                output["relation_mask"],
            )
            projected = teacher.relation_align_head(pooled)
            valid_rows = output["relation_mask"].any(dim=-1)
            aux = relation_alignment_loss(
                projected,
                output["source_embedding"],
                valid_rows,
            )
            loss = loss + relation_aux_weight * aux
        # 官方 TGN 顺序：本批预测完成后才写入真实正边。
        teacher.update_state(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), grad_clip)
        optimizer.step()
        teacher.detach_state()
        meter.update(float(loss.detach()), int(batch["target"].numel()))
    return meter.average


def _copy_state_dict(source) -> Dict[str, Tensor]:
    state = source.state_dict() if isinstance(source, nn.Module) else source
    return {
        name: value.detach().cpu().clone() for name, value in state.items()
    }


def _make_ema(model: nn.Module, decay: float) -> Optional[Dict[str, Tensor]]:
    """创建学生权重的 EMA 影子副本；decay<=0 时返回 None（不启用）。"""

    if decay <= 0.0:
        return None
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _ema_update(
    shadow: Optional[Dict[str, Tensor]],
    model: nn.Module,
    decay: float,
) -> None:
    if shadow is None:
        return
    for name, value in model.state_dict().items():
        slot = shadow[name]
        if value.dtype.is_floating_point:
            slot.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            slot.copy_(value)


def _evaluate_state(
    model: nn.Module,
    state: Dict[str, Tensor],
    bundle: TemporalGraphBundle,
    loader: CandidateBatchLoader,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    """临时把 model 权重换成 state 做一次评测，随后恢复原权重。"""

    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state, strict=True)
    result = evaluate_split(model, bundle, loader, split, args, device)
    model.load_state_dict(backup, strict=True)
    return result


def _metric_value(metrics: Dict[str, object]) -> float:
    metric_name = str(metrics["metric"])
    return float(metrics[metric_name])


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        if patience < 0:
            raise ValueError("patience must be non-negative.")
        self.patience = patience
        self.best_metric = float("-inf")
        self.best_epoch = 0
        self.bad_epochs = 0

    def update(self, metric: float, epoch: int) -> tuple[bool, bool]:
        improved = metric > self.best_metric
        if improved:
            self.best_metric = metric
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        should_stop = self.patience > 0 and self.bad_epochs >= self.patience
        return improved, should_stop


def print_early_stopping(
    name: str,
    epoch: int,
    stopper: EarlyStopping,
    metric_prefix: str,
    metric_name: str,
) -> None:
    print(
        f"Early stopping {name} at epoch {epoch:02d}: "
        f"no validation improvement for {stopper.bad_epochs} epoch(s), "
        f"best_epoch={stopper.best_epoch:02d}, "
        f"best_val_{metric_prefix}_{metric_name}={stopper.best_metric:.4f}, "
        f"patience={stopper.patience}"
    )


def pretrain_teacher(
    teacher: nn.Module,
    bundle: TemporalGraphBundle,
    train_loader: CandidateBatchLoader,
    validation_loader: CandidateBatchLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Tensor]:
    optimizer = torch.optim.AdamW(
        teacher.parameters(),
        lr=args.teacher_lr,
        weight_decay=args.weight_decay,
    )
    best_metric = -1.0
    best_state: Optional[Dict[str, Tensor]] = None
    stopper = EarlyStopping(args.patience)
    for epoch in range(1, args.teacher_epochs + 1):
        train_loss = train_teacher_epoch(
            teacher,
            train_loader,
            optimizer,
            device,
            args.grad_clip,
            relation_aux_weight=float(args.teacher_relation_weight or 0.0),
        )
        validation = evaluate_split(
            teacher,
            bundle,
            validation_loader,
            "val",
            args,
            device,
        )
        metric_name = str(validation["metric"])
        prefix = "official" if validation["official"] else "smoke"
        metric_value = _metric_value(validation)
        print(
            f"Teacher {epoch:02d}/{args.teacher_epochs:02d} "
            f"train_loss={train_loss:.4f} "
            f"val_{prefix}_{metric_name}={metric_value:.4f}"
        )
        improved, should_stop = stopper.update(metric_value, epoch)
        if improved:
            best_metric = metric_value
            best_state = _copy_state_dict(teacher)
        if should_stop:
            print_early_stopping("Teacher", epoch, stopper, prefix, metric_name)
            break
    if best_state is None:
        raise RuntimeError("Teacher training did not produce a checkpoint.")
    teacher.load_state_dict(best_state, strict=True)
    teacher.reset_state()
    return best_state


def train_student_epoch(
    student: LightGNNStudent,
    teacher: nn.Module,
    objective: DistillationObjective,
    loader: CandidateBatchLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    ema_shadow: Optional[Dict[str, Tensor]] = None,
    ema_decay: float = 0.0,
    mosa_rkd_weight: float = 0.0,
) -> Dict[str, float]:
    student.train()
    teacher.eval()
    # 仅在确实需要关系张量时才请求，避免零权重分支的额外开销。
    need_teacher_relation = (
        objective.anchor_weight != 0.0
        or getattr(objective, "kernel_weight", 0.0) != 0.0
    )
    need_student_relation = (
        objective.diversity_weight != 0.0
        or objective.anchor_weight != 0.0
    )
    # 有状态学生（轻量 GNN）需要按 TGN 顺序先打分后写入正边。
    update_student_state = getattr(student, "update_state", None)
    teacher_scorer = getattr(teacher, "score_candidates", None)
    meters: Dict[str, AverageMeter] = {}
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        with torch.no_grad():
            if callable(teacher_scorer):
                teacher_output = teacher_scorer(
                    batch,
                    return_relations=need_teacher_relation,
                )
            else:
                teacher_output = teacher(
                    batch,
                    return_relations=need_teacher_relation,
                )
            teacher.update_state(batch)
        student_output = student(
            batch,
            return_relations=need_student_relation,
        )
        if callable(update_student_state):
            update_student_state(batch)
        optimizer.zero_grad(set_to_none=True)
        total, components = objective(
            student_output,
            teacher_output,
            batch["target"],
        )
        if mosa_rkd_weight != 0.0:
            if not isinstance(teacher, EnsembleStudent):
                raise ValueError(
                    "--mosa-rkd-weight requires a MOSA student ensemble."
                )
            teacher_candidates = getattr(
                teacher, "last_candidate_embeddings", []
            )
            if not teacher_candidates:
                raise ValueError(
                    "MOSA student ensemble did not expose candidate embeddings."
                )
            student_candidates = student_output["candidate_embeddings"]
            per_teacher = [
                rkd_distance_loss(student_candidates, target.detach())
                + rkd_angle_loss(student_candidates, target.detach())
                for target in teacher_candidates
            ]
            mosa_rkd = torch.stack(per_teacher).mean()
            total = total + mosa_rkd_weight * mosa_rkd
            components = dict(components)
            components["mosa_rkd"] = mosa_rkd.detach()
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        optimizer.step()
        _ema_update(ema_shadow, student, ema_decay)
        count = int(batch["target"].numel())
        for name, value in components.items():
            meters.setdefault(name, AverageMeter()).update(float(value), count)
    return {name: meter.average for name, meter in meters.items()}


def distill_student(
    student: nn.Module,
    teacher: nn.Module,
    bundle: TemporalGraphBundle,
    train_loader: CandidateBatchLoader,
    validation_loader: CandidateBatchLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Tensor]:
    if args.method == "trd":
        objective = TemporalDeltaDistillation(
            args.logit_weight, args.trd_weight, args.logit_temperature, args.trd_mode,
        ).to(device)
    else:
        objective = DistillationObjective(
            task_weight=args.task_weight,
            logit_weight=args.logit_weight,
            rank_weight=args.rank_weight,
            diversity_weight=args.diversity_weight,
            anchor_weight=float(args.anchor_weight or 0.0),
            kernel_weight=float(args.kernel_weight or 0.0),
            kernel_divergence=str(args.kernel_divergence),
            kernel_teacher_mode=str(args.kernel_teacher_mode),
            rkd_weight=float(args.rkd_weight),
            rkd_mode=str(args.rkd_mode),
            confidence_weight=float(args.confidence_weight),
            dkd_weight=float(args.dkd_weight),
            dkd_tckd_weight=float(args.dkd_tckd_weight),
            dkd_nckd_weight=float(args.dkd_nckd_weight),
            logit_temperature=args.logit_temperature,
            rank_temperature=args.rank_temperature,
        ).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.student_lr,
        weight_decay=args.weight_decay,
    )
    best_metric = -1.0
    best_state: Optional[Dict[str, Tensor]] = None
    stopper = EarlyStopping(args.patience)
    reset_student_state = getattr(student, "reset_state", None)
    ema_decay = float(getattr(args, "ema_decay", 0.0))
    ema_shadow = _make_ema(student, ema_decay)
    for epoch in range(1, args.student_epochs + 1):
        teacher.reset_state()
        if callable(reset_student_state):
            reset_student_state()
        if args.method == "trd":
            losses = train_temporal_delta_epoch(
                student, teacher, objective, train_loader, optimizer, device,
                args.grad_clip, ema_callback=lambda: _ema_update(ema_shadow, student, ema_decay),
            )
        else:
            losses = train_student_epoch(
                student,
                teacher,
                objective,
                train_loader,
                optimizer,
                device,
                args.grad_clip,
                ema_shadow=ema_shadow,
                ema_decay=ema_decay,
                mosa_rkd_weight=float(
                    getattr(args, "mosa_rkd_weight", 0.0)
                ),
            )
        if ema_shadow is not None:
            validation = _evaluate_state(
                student,
                ema_shadow,
                bundle,
                validation_loader,
                "val",
                args,
                device,
            )
        else:
            validation = evaluate_split(
                student,
                bundle,
                validation_loader,
                "val",
                args,
                device,
            )
        metric_name = str(validation["metric"])
        prefix = "official" if validation["official"] else "smoke"
        metric_value = _metric_value(validation)
        loss_text = " ".join(f"{name}={value:.6f}" for name, value in losses.items())
        print(f"Student {epoch:02d}/{args.student_epochs:02d} {loss_text} "
              f"val_{prefix}_{metric_name}={metric_value:.4f}")
        improved, should_stop = stopper.update(metric_value, epoch)
        if improved:
            best_metric = metric_value
            best_state = _copy_state_dict(
                ema_shadow if ema_shadow is not None else student
            )
        if should_stop:
            print_early_stopping("Student", epoch, stopper, prefix, metric_name)
            break
    if best_state is None:
        raise RuntimeError("Student distillation did not produce a checkpoint.")
    student.load_state_dict(best_state, strict=True)
    return best_state


def train_supervised_student_epoch(
    student: nn.Module,
    loader: CandidateBatchLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    ema_shadow: Optional[Dict[str, Tensor]] = None,
    ema_decay: float = 0.0,
) -> float:
    student.train()
    update_student_state = getattr(student, "update_state", None)
    meter = AverageMeter()
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = student(batch, return_relations=False)
        if callable(update_student_state):
            update_student_state(batch)
        loss = F.cross_entropy(output["logits"], batch["target"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        optimizer.step()
        _ema_update(ema_shadow, student, ema_decay)
        meter.update(float(loss.detach()), int(batch["target"].numel()))
    return meter.average


def train_supervised_student(
    student: nn.Module,
    bundle: TemporalGraphBundle,
    train_loader: CandidateBatchLoader,
    validation_loader: CandidateBatchLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Tensor]:
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.student_lr,
        weight_decay=args.weight_decay,
    )
    best_metric = -1.0
    best_state: Optional[Dict[str, Tensor]] = None
    stopper = EarlyStopping(args.patience)
    reset_student_state = getattr(student, "reset_state", None)
    ema_decay = float(getattr(args, "ema_decay", 0.0))
    ema_shadow = _make_ema(student, ema_decay)
    for epoch in range(1, args.student_epochs + 1):
        if callable(reset_student_state):
            reset_student_state()
        train_loss = train_supervised_student_epoch(
            student,
            train_loader,
            optimizer,
            device,
            args.grad_clip,
            ema_shadow=ema_shadow,
            ema_decay=ema_decay,
        )
        if ema_shadow is not None:
            validation = _evaluate_state(
                student,
                ema_shadow,
                bundle,
                validation_loader,
                "val",
                args,
                device,
            )
        else:
            validation = evaluate_split(
                student,
                bundle,
                validation_loader,
                "val",
                args,
                device,
            )
        metric_name = str(validation["metric"])
        prefix = "official" if validation["official"] else "smoke"
        metric_value = _metric_value(validation)
        print(
            f"Supervised student {epoch:02d}/{args.student_epochs:02d} "
            f"train_loss={train_loss:.4f} "
            f"val_{prefix}_{metric_name}={metric_value:.4f}"
        )
        improved, should_stop = stopper.update(metric_value, epoch)
        if improved:
            best_metric = metric_value
            best_state = _copy_state_dict(
                ema_shadow if ema_shadow is not None else student
            )
        if should_stop:
            print_early_stopping(
                "Supervised student",
                epoch,
                stopper,
                prefix,
                metric_name,
            )
            break
    if best_state is None:
        raise RuntimeError("Supervised student training did not produce a checkpoint.")
    student.load_state_dict(best_state, strict=True)
    return best_state


def freeze_teacher(teacher: nn.Module) -> None:
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)


def print_metrics(name: str, metrics: Dict[str, object]) -> None:
    metric_name = str(metrics["metric"])
    prefix = "official" if metrics["official"] else "smoke"
    details = f"{prefix}_{metric_name}={_metric_value(metrics):.4f}"
    if "loss" in metrics:
        details += f", loss={float(metrics['loss']):.4f}"
    if "hits@1" in metrics:
        details += f", hits@1={float(metrics['hits@1']):.4f}"
    print(f"{name}: {details}")


def print_latency(name: str, latency: Dict[str, float]) -> None:
    print(
        f"{name} latency: mean={latency['batch_ms_mean']:.3f} ms/batch, "
        f"p95={latency['batch_ms_p95']:.3f} ms/batch, "
        f"throughput={latency['queries_per_second']:.1f} query/s"
    )


def print_model_parameters(name: str, model: nn.Module) -> None:
    summary = model_parameter_summary(model)
    print(
        f"{name} parameters: "
        f"trainable={summary['trainable']:,}, "
        f"all={summary['all']:,}, "
        f"runtime_state_elements={summary['runtime_state_elements']:,}, "
        f"total_with_runtime_state={summary['total_with_runtime_state']:,}"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.quick:
        apply_quick_configuration(args)
    apply_method_configuration(args)
    resolve_student_seed(args)
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Arguments: {json.dumps(vars(args), ensure_ascii=True, sort_keys=True)}")

    bundle, train_loader, validation_loader, test_loader = make_loaders(args)
    data_signature = compute_data_signature(args, bundle)
    print(f"Data signature: {data_signature}")
    print(
        "Evaluation protocol: "
        + ("official TGB negatives" if bundle.official_evaluation else "smoke candidates")
    )
    teacher: Optional[nn.Module] = None
    teacher_metrics: Optional[Dict[str, object]] = None

    if args.stage in {"teacher", "all"}:
        teacher = build_teacher(bundle, args).to(device)
        print_model_parameters("Teacher", teacher)
        pretrain_teacher(
            teacher,
            bundle,
            train_loader,
            validation_loader,
            args,
            device,
        )
        teacher_path = output_dir / "teacher.pt"
        save_model_checkpoint(
            teacher_path,
            teacher,
            model_name=type(teacher).__name__,
            model_config={**teacher.model_config, "data_signature": data_signature},
            args=vars(args),
        )
        print(f"Saved teacher checkpoint: {teacher_path}")
        teacher_metrics = evaluate_teacher_from_history(
            teacher,
            bundle,
            [train_loader, validation_loader],
            test_loader,
            "test",
            args,
            device,
        )
        print_metrics("Teacher test", teacher_metrics)

    if args.stage == "distill":
        ensemble_paths = [
            item.strip()
            for item in str(getattr(args, "teacher_checkpoints", "")).split(",")
            if item.strip()
        ]
        mosa_paths = [
            item.strip()
            for item in str(
                getattr(args, "mosa_teacher_checkpoints", "")
            ).split(",")
            if item.strip()
        ]
        if mosa_paths:
            students = [
                load_student_from_checkpoint(path, bundle, device, data_signature=data_signature)
                for path in mosa_paths
            ]
            teacher = EnsembleStudent(students).to(device)
            print("Loaded MOSA student ensemble as teacher:")
            for path in mosa_paths:
                print(f"  - {path}")
        elif ensemble_paths:
            teachers = [
                load_teacher_from_checkpoint(
                    path,
                    bundle,
                    data_signature,
                    device,
                    temporal_neighbors=(
                        args.teacher_query_neighbors
                        if int(getattr(args, "teacher_query_neighbors", 0)) > 0
                        else None
                    ),
                )
                for path in ensemble_paths
            ]
            teacher = EnsembleTeacher(teachers).to(device)
            print("Loaded teacher ensemble:")
            for path in ensemble_paths:
                print(f"  - {path}")
        else:
            teacher = load_teacher_from_checkpoint(
                args.teacher_checkpoint,
                bundle,
                data_signature,
                device,
                temporal_neighbors=(
                    args.teacher_query_neighbors
                    if int(getattr(args, "teacher_query_neighbors", 0)) > 0
                    else None
                ),
            )
            print(f"Loaded teacher checkpoint: {args.teacher_checkpoint}")
        teacher_layers = getattr(teacher, "teacher_layers", None)
        if args.teacher_hidden != teacher.hidden_dim or (
            teacher_layers is not None and args.teacher_layers != teacher_layers
        ):
            print(
                "Warning: loaded teacher config overrides CLI args: "
                f"hidden_dim={teacher.hidden_dim} (arg {args.teacher_hidden}), "
                f"teacher_layers={teacher_layers} "
                f"(arg {args.teacher_layers})."
            )
        print_model_parameters("Teacher", teacher)

    if args.stage in {"distill", "all"}:
        if teacher is None:
            raise RuntimeError("Distillation requires an initialized teacher.")
        if args.anchor_weight and int(args.relation_dim) != int(teacher.relation_dim):
            raise ValueError(
                "Student relation_dim must match the teacher relation_dim: "
                f"got {args.relation_dim}, expected {teacher.relation_dim}."
            )
        # 学生独立种子：不影响已构造的数据/教师，仅控制学生初始化与 dropout。
        seed_everything(int(args.student_seed))
        freeze_teacher(teacher)
        student = build_student(bundle.node_features, args, bundle.structure_features, bundle=bundle).to(device)
        print_model_parameters("Student", student)
        distill_student(
            student,
            teacher,
            bundle,
            train_loader,
            validation_loader,
            args,
            device,
        )
        student_path = output_dir / "student.pt"
        save_model_checkpoint(
            student_path,
            student,
            model_name=type(student).__name__,
            model_config={**student.model_config, "data_signature": data_signature},
            args=vars(args),
        )
        print(f"Saved student checkpoint: {student_path}")

        if teacher_metrics is None:
            teacher_metrics = evaluate_teacher_from_history(
                teacher,
                bundle,
                [train_loader, validation_loader],
                test_loader,
                "test",
                args,
                device,
            )
            print_metrics("Teacher test", teacher_metrics)
        student_metrics = evaluate_teacher_from_history(
            student,
            bundle,
            [train_loader, validation_loader],
            test_loader,
            "test",
            args,
            device,
        )
        print_metrics("Student test", student_metrics)

        teacher.eval()
        teacher.reset_state()
        replay_teacher(teacher, [train_loader, validation_loader], device)
        student.reset_state()
        replay_teacher(student, [train_loader, validation_loader], device)
        warmup_batches = 1 if args.quick else 5
        # 只有 TGN 教师支持 update_memory；LightST 教师直接 forward。
        teacher_forward_kwargs = (
            {"update_memory": False}
            if isinstance(
                teacher,
                (EnsembleTeacher, PyGTGNTeacher, GraphMixerTeacher, TGATTeacher),
            )
            else {}
        )
        teacher_latency = benchmark_latency(
            teacher,
            test_loader,
            device,
            warmup_batches=warmup_batches,
            measured_batches=args.latency_batches,
            forward_kwargs=teacher_forward_kwargs,
        )
        student_latency = benchmark_latency(
            student,
            test_loader,
            device,
            warmup_batches=warmup_batches,
            measured_batches=args.latency_batches,
        )
        print_latency("Teacher", teacher_latency)
        print_latency("Student", student_latency)
        speedup = teacher_latency["query_ms"] / max(
            student_latency["query_ms"],
            1.0e-12,
        )
        print(f"Model-only query latency speedup: {speedup:.2f}x")

    if args.stage == "student":
        seed_everything(int(args.student_seed))
        student = build_student(bundle.node_features, args, bundle.structure_features, bundle=bundle).to(device)
        print_model_parameters("Student", student)
        train_supervised_student(
            student,
            bundle,
            train_loader,
            validation_loader,
            args,
            device,
        )
        student_path = output_dir / "student_supervised.pt"
        save_model_checkpoint(
            student_path,
            student,
            model_name=type(student).__name__,
            model_config={**student.model_config, "data_signature": data_signature},
            args=vars(args),
        )
        print(f"Saved supervised student checkpoint: {student_path}")
        print_metrics(
            "Supervised student test",
            evaluate_teacher_from_history(
                student,
                bundle,
                [train_loader, validation_loader],
                test_loader,
                "test",
                args,
                device,
            ),
        )
        student.reset_state()
        replay_teacher(student, [train_loader, validation_loader], device)
        student_latency = benchmark_latency(
            student,
            test_loader,
            device,
            warmup_batches=1 if args.quick else 5,
            measured_batches=args.latency_batches,
        )
        print_latency("Student", student_latency)


if __name__ == "__main__":
    main()
