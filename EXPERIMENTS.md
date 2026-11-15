# TRD 实验记录（GitHub 快照）

快照日期：2026-11-15。当前 Wiki 主矩阵仍在服务器后台运行；本文件是快照，实时状态见 `results/STATUS.md`。

## 环境

- Python 3.10.19（conda `Graph` 环境）
- torch 2.1.0+cu118，torch-geometric 2.6.1，numpy 1.26.4
- py-tgb 2.2.0，通过 `PYTHONPATH=/data/ljh/RKD/.benchmark-deps` 加载
- GPU：2 × RTX 4090

## 数据

五个 TGB 数据集已完成下载、zip 校验、官方切分和负例文件校验。数据本身不进入 Git（约 40 GB）。

| 数据集 | 节点 | 事件 | train / val / test |
|---|---:|---:|---|
| tgbl-wiki v2 | 9,227 | 157,474 | 110,232 / 23,621 / 23,621 |
| tgbl-review v2 | 352,637 | 4,873,540 | 3,413,837 / 730,784 / 728,919 |
| tgbl-coin v2 | 638,486 | 22,809,486 | 15,966,659 / 3,421,416 / 3,421,411 |
| tgbl-comment v1 | 994,790 | 44,314,507 | 31,020,155 / 6,647,176 / 6,647,176 |
| tgbl-flight v2 | 18,143 | 67,169,570 | 47,043,612 / 10,099,015 / 10,026,943 |

数据清单与指纹：`results/{wiki,review,coin,comment,flight}-data.json`；zip/负例 SHA256：`results/data-sha256.txt`。

## 复现命令

```bash
python -m pip install -r requirements-benchmark.txt
PYTHONPATH=.benchmark-deps python -m experiments.prepare_data \
  --datasets tgbl-wiki --data-root datasets --download --output results/wiki-data.json
PYTHONPATH=.benchmark-deps python -m experiments.matrix \
  --config configs/smoke_cuda.json --output-root results/smoke-cuda-check --execute
PYTHONPATH=.benchmark-deps python -m experiments.matrix \
  --config configs/comparison.json --output-root results/main --execute
PYTHONPATH=.benchmark-deps python -m experiments.summarize \
  results/main --output results/main/summary.csv
```

消融脚本：`run_trd_ablation.py`、`run_capacity_ablation.py`。

## 当前 Wiki 结果（seed42，官方 test MRR）

### GCN 学生

| 方法 | TGN | TGAT | GraphMixer |
|---|---:|---:|---:|
| 监督 GCN | 0.4373 | — | — |
| Logits KD | 0.4600 | 0.4680 | 0.4726 |
| TRD | 0.4596 | 0.4637 | 0.4632 |
| RKD-distance | 0.4700 | 0.4725 | 0.4678 |
| RKD-angle | 0.4646 | 0.4709 | 0.4725 |
| DKD | 0.4473 | 0.4445 | 0.4701 |
| Logit Standardization | 0.4485 | 0.4488 | 0.4490 |
| NOSMOG-RSD component | 0.2293 | 0.4189 | 0.4433 |

### SAGE 学生（进行中）

- 监督 SAGE：0.6535
- SAGE + TGN + Logits KD：0.6697
- SAGE + TGN + DKD：0.6661
- 其余方法仍在运行。

### TRD 消融（GCN + TGN + seed42）

| 变体 | Test MRR |
|---|---:|
| temporal（主矩阵） | 0.4596 |
| static | 0.4608 |
| shuffled | 0.4470 |
| reverse | 0.4669 |
| zero / kd0 / rel0 | 运行/排队中 |

以上均为单 seed 阶段性结果，不能作为最终结论。`shuffled` 下降说明配对关系有作用；`reverse` 在单 seed 下高于 temporal，因此当前不能声称时序增量优于控制项。

## 边界与未完成项

- Wiki 141-run 矩阵仍在运行（快照时完成 28 个 run），最终 3-seed × 3 教师 × 2 学生汇总尚未生成。
- Review/Coin/Comment/Flight 的全量矩阵在当前 2×4090 上不可行：Review 单 epoch TGN 实测约 6308 秒，Coin 事件量是 Wiki 的 145 倍，Comment 281 倍，Flight 427 倍。
- 未虚构任何未运行的实验数据；未修改 TRD 公式或实验协议。
