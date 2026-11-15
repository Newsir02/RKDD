# TRD

Temporal Relation Delta Distillation（TRD）用于连续时间图上的链接预测。时序教师为同一个查询计算两个历史前缀下的节点关系，普通 GNN 学生学习两个关系值之间的有符号增量。训练目标由任务交叉熵、温度 Logits KL 和关系增量 MSE 组成。

## 论文基线与统一比较入口

新增 `benchmark.py` 支持 TGN/TGAT/GraphMixer 教师、普通 GCN/GraphSAGE 学生、Logits KD、DKD、RKD-D/A、Logit Standardization、NOSMOG-RSD 组件适配及原 TRD。原训练入口继续保留。论文出处、完整复现与组件适配的区别见 [distillation_methods.md](distillation_methods.md)；五数据集、三种子、705 次运行计划及公平性协议见 [comparison_experiments.md](comparison_experiments.md)。验证证据见 [validation.md](docs/research/validation.md)。

```bash
python -m experiments.matrix --config configs/smoke.json --output-root results/my-smoke --execute
python -m experiments.summarize results/my-smoke --output results/my-smoke/summary.csv
```

以上是合成数据验证。真实五数据集的完整性能结果尚未生成。

## 环境

本地使用 Python 3.10、PyTorch 2.1.0 和 PyG 2.6.1 验证。可复用具备这些依赖的 Python 环境，也可在新环境安装。CUDA 环境请按服务器驱动安装相应的 PyTorch 构建。

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
```

加载 TGB 数据以及运行完整 TGB/UCI 评测时，安装项目使用的 [py-tgb 2.2.0](https://pypi.org/project/py-tgb/2.2.0/)。合成数据训练无需此依赖。

```bash
python -m pip install py-tgb==2.2.0
```

## 数据

项目包含全部数据加载器和内置合成数据生成器。真实数据文件通过 `--data-root` 指定，默认目录为 `datasets/`。

| `--dataset` | 准备方式 |
|---|---|
| `synthetic` | 程序按种子生成事件、特征与时间划分，可离线运行 |
| `tgbl-wiki`、`tgbl-subreddit` 等受支持的 `tgbl-*` | 通过 PyG TGB 包装器加载，默认允许下载。已有数据可指定 `--data-root` 和 `--no-tgb-download`。 [TGB 链接预测文档](https://tgb.complexdatalab.com/docs/linkprop/) |
| `tgbl-enron` | 在 `datasets/tgbl_enron/` 放置 `tgbl-enron_edgelist.csv`、`tgbl-enron_val_ns.pkl`、`tgbl-enron_test_ns.pkl`。使用与该数据版本配套的负样本文件。 [TGB 数据说明](https://tgb.complexdatalab.com/docs/dataset_overview/) |
| `uci` | 在 `datasets/uci/` 放置预处理后的 `ml_uci.csv`、`ml_uci.npy`、`ml_uci_node.npy`。CSV 需含 `u,i,ts`，边特征对应 CSV 行，节点特征按节点 ID 索引。按时间 70/15/15 划分，验证和测试使用固定采样负例。 [UCI 原始交互数据](https://snap.stanford.edu/data/CollegeMsg.html) |

真实数据和训练产物需自行准备，仓库可直接生成合成数据。`--quick` 对真实数据只运行小前缀，用于接口检查。正式评测使用完整数据和对应的评测负例。

## 运行 TRD

在项目目录运行。CPU 小规模训练命令

```bash
python train_trd.py --stage all --dataset synthetic --quick --device cpu --batch-size 12 --output-dir checkpoints/smoke-trd
```

正式实验可以先训练教师，再复用同一 checkpoint。`--teacher-arch` 支持 `tgn`、`tgat`、`graphmixer`。

```bash
python train_trd.py --stage teacher --dataset tgbl-wiki --device cuda --teacher-arch tgn --seed 42 --output-dir checkpoints/wiki-teacher
python train_trd.py --stage distill --dataset tgbl-wiki --device cuda --seed 42 --student-seed 42 --teacher-checkpoint checkpoints/wiki-teacher/teacher.pt --logit-weight 1 --trd-weight 1 --output-dir checkpoints/wiki-trd
```

教师和学生 checkpoint 分别保存为 `teacher.pt` 和 `student.pt`。监督训练输出 `student_supervised.pt`。加载教师时检查数据指纹。复用 checkpoint 时，沿用数据集、数据版本、数据种子和候选数等配置。

## 对比方法

同一普通 GNN 的对照通过 `train_trd.py` 运行。除监督基线外，使用 `--stage distill` 和相同教师 checkpoint。

| 方法 | 参数 |
|---|---|
| 监督 GNN | `--stage student` |
| Logits KD | `--trd-weight 0` |
| 静态余弦关系 | `--trd-mode static` |
| RKD 距离／角度 | `--trd-mode rkd_distance` / `--trd-mode rkd_angle` |
| TRD | `--trd-mode temporal` |
| TRD 无 Logits KD | `--logit-weight 0 --trd-mode temporal` |
| TRD 消融 | `--trd-mode shuffled` / `reverse` / `zero` |

例如，使用上面的正式教师训练 Logits KD

```bash
python train_trd.py --stage distill --dataset tgbl-wiki --device cuda --seed 42 --student-seed 42 --teacher-checkpoint checkpoints/wiki-teacher/teacher.pt --trd-weight 0 --output-dir checkpoints/wiki-logit-kd
```

`train.py` 运行沿用 `LightGNNStudent` 的对照。公共参数为 `--stage distill --dataset tgbl-wiki --teacher-checkpoint checkpoints/wiki-teacher/teacher.pt`，再附加选定方法和独立的 `--output-dir`。

| 方法 | 附加参数 |
|---|---|
| Logits + 排序 KD | `--method baseline` |
| DKD | `--method dkd` |
| TRID | `--method trid --student-aggregation kernel --rank-weight 0` |

集成评测使用 `python eval_ensemble_students.py --dataset tgbl-wiki --checkpoints path1.pt,path2.pt`。多教师 KD 和 MOSA-S 可分别向 `train.py` 传入 `--teacher-checkpoints` 和 `--mosa-teacher-checkpoints`。

## 代码结构

| 位置 | 内容 |
|---|---|
| `train_trd.py`、`train.py` | TRD 入口、公共训练和评测、对照方法入口 |
| `model/` | 三种时序教师、普通均值聚合学生、对照学生与集成 |
| `loss/` | 三项 TRD 目标及对照损失 |
| `util/temporal_delta_training.py` | 同查询双历史前缀训练 |
| `dataset/` | 合成流、TGB、Enron、UCI 和候选加载 |
| `tests/` | 模型、损失、数据和 checkpoint 回归测试 |

有 `py-tgb` 时运行 `python -m pytest -q`。没有该依赖时可运行其余回归测试：

```bash
python -m pytest -q --ignore=tests/test_tgb_dependency_compatibility.py --deselect tests/test_tgb_evaluation.py::test_official_evaluation_queries_negatives_then_updates_teacher --deselect tests/test_tgb_evaluation.py::test_official_test_split_loads_test_negatives
```

方法定义、适用条件和实验设计见 [方法研究](METHOD_RESEARCH.md)。
