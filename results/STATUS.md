# TRD 实验运行状态

更新时间：2026-11-14 18:00 左右（服务器本地时间）

## 环境

- 解释器：`/home/ubuntu/anaconda3/envs/Graph/bin/python`（Python 3.10.19）
- torch 2.1.0+cu118，PyG 2.6.1，numpy 1.26.4
- py-tgb 2.2.0 通过 `PYTHONPATH=/data/ljh/RKD/.benchmark-deps` 使用
- GPU：2 × RTX 4090；本次矩阵使用 `CUDA_VISIBLE_DEVICES=1`
- 数据与结果目录：`/data/ljh/RKD/{datasets,results}`

## 已完成的验证

- 完整回归：`126 passed in 9.01s`（Graph 环境，pytest 9.2.0）
- 合成 smoke 矩阵：141/141 runs 成功，0 failure，47 个 3-seed 分组全部 complete
- 真实 Wiki 调试（1 epoch）：
  - 监督 GCN val MRR 0.4398
  - Logits KD GCN val MRR 0.4624
  - TRD GCN val MRR 0.4420
  - TRD SAGE val MRR 0.6948
  - 以上均为 1-epoch 调试验证，不作为正式结果

### 其余 baseline 的真实数据 1-epoch 调试（GCN，seed42，TGN 教师）

| 方法 | 官方 val MRR | 1 epoch 训练秒 | 备注 |
|---|---:|---:|---|
| DKD | 0.4575 | 51.7 | 正常 |
| RKD-distance | 0.4603 | 354.9 | 查询内两两距离循环，明显慢 |
| RKD-angle | 0.4642 | 163.4 | 查询内两两角度循环，明显慢 |
| Logit Standardization | 0.4494 | 47.5 | 正常 |
| NOSMOG-RSD component | 0.0850 | 47.7 | 未归一化 Gram MSE 量级约 3.5e3，1 epoch 后仍需观察；按文档不修改公式 |

以上均为 1-epoch 调试验证，不作为正式结果。RKD 的正式 run 预计显著慢于其他方法。

### Review 真实规模验证（重要）

- 数据：352,637 节点 / 4,873,540 事件；官方 train 3,413,837 / val 730,784 / test 728,919。
- 1-epoch TGN 教师：训练 **6308 秒（约 1.75 小时）**，官方 val MRR 0.2979。
- 3 小时超时前已完成训练与 val，但在 test 历史重放/评估阶段被中断，未写出 result.json。
- 结论：Review 单个教师 100 epochs 约需一周以上；141-run Review 矩阵在当前 2×4090 上不可行。该结论是实测，不是推测。当前只继续 Wiki 全矩阵。

### 其余数据集

- Coin：22,809,486 事件，约为 Wiki 的 145 倍；即使按 Wiki TGN 的 36 秒/epoch 线性下限，也约 1.5 小时/epoch，实际更慢。单个教师 100 epochs 需要数周。
- Comment：44,314,507 事件（281 倍 Wiki）；Flight：67,169,570 事件（427 倍 Wiki）。
- 因此 705-run 全量矩阵在当前 2×4090 上不可行。后续数据集只做下载、官方切分/负例校验和数据指纹记录；不启动无法完成的正式训练，避免产生半成品或虚构结果。

## 数据集状态

| 数据集 | 节点 | 事件 | 官方切分 | 状态 |
|---|---:|---:|---|---|
| tgbl-wiki v2 | 9,227 | 157,474 | 110,232 / 23,621 / 23,621 | 已下载、已校验、已生成 data signature |
| tgbl-review v2 | 352,637 | 4,873,540 | 3,413,837 / 730,784 / 728,919 | 已下载、已校验 |
| tgbl-coin v2 | 638,486 | 22,809,486 | 15,966,659 / 3,421,416 / 3,421,411 | 已下载校验，data signature `a59d50be...` |
| tgbl-comment v1 | 994,790 | 44,314,507 | 31,020,155 / 6,647,176 / 6,647,176 | 已下载校验，data signature `1bdc0013...` |
| tgbl-flight v2 | 18,143 | 67,169,570 | 47,043,612 / 10,099,015 / 10,026,943 | 已下载校验，data signature `511cc351...` |

全部 5 个数据集均已完成下载、zip CRC/尺寸校验、官方切分与负例文件校验；数据清单见 `results/*-data.json`，全部 zip 和负例文件 SHA256 见 `results/data-sha256.txt`。

数据清单：`results/wiki-data.json`、`results/review-data.json`
压缩包/负例文件哈希：`results/data-sha256.txt`

## Wiki 正式矩阵（进行中）

命令：`python -m experiments.matrix --config configs/comparison.json --output-root results/main --execute --max-runs 141`
输出：`results/main/tgbl-wiki/...`，日志：`results/main-wiki.log`

已完成：

| 教师/学生 | seed | 官方 test MRR | best epoch / epochs run | 训练秒 |
|---|---:|---:|---:|---:|
| TGN 教师 | 42 | 0.6114 | 22 / 42 | 1539 |
| TGAT 教师 | 42 | 0.6450 | 2 / 22 | 423 |
| GraphMixer 教师 | 42 | 0.6654 | 2 / 22 | 360 |
| GCN 监督学生 | 42 | 0.4373 | 5 / 25 | 274 |
| GCN + Logits KD (TGN) | 42 | 0.4600 | 21 / 41 | 847 |
| GCN + DKD (TGN) | 42 | 0.4473 | 5 / 25 | 585 |
| GCN + RKD-distance (TGN) | 42 | 0.4700 | 35 / 55 | 9484 |
| GCN + RKD-angle (TGN) | 42 | 0.4646 | 15 / 35 | 4101 |
| GCN + Logit Standardization (TGN) | 42 | 0.4485 | 5 / 25 | 519 |
| GCN + NOSMOG-RSD (TGN) | 42 | 0.2293 | 30 / 50 | 1046 |
| GCN + TRD (TGN) | 42 | 0.4596 | 14 / 34 | 1032 |

当前正在运行：SAGE / TGN / RKD-distance（SAGE 的 DKD=0.6661、Logits KD=0.6697 已完成）。之后继续 SAGE 其余方法，再到 seeds 43/44。

### Wiki GCN + TGN seed42 初步对比（非最终）

| 方法 | Test MRR |
|---|---:|
| Logits KD | 0.4600 |
| TRD | 0.4596 |
| RKD-distance | 0.4700 |
| RKD-angle | 0.4646 |
| Logit Standardization | 0.4485 |
| DKD | 0.4473 |
| 监督 GCN | 0.4373 |
| NOSMOG-RSD component | 0.2293 |

单 seed、单教师、单学生，不能作为最终结论。NOSMOG-RSD 的未归一化 Gram 目标量级过大，训练不稳定；按文档未改公式。TRD 与 Logits KD 接近，低于 RKD-distance/angle；需要完整 3 教师 × 2 学生 × 3 seeds 后再判断。

### Wiki GCN seed42：三个教师（初步，非最终）

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

GCN seed42 已完成全部 3 教师 × 7 方法。SAGE 监督学生已完成：官方 test MRR **0.6535**（best epoch 1，21 epochs）；SAGE + TGN + Logits KD 已完成：**0.6697**（best epoch 3，23 epochs）。SAGE 明显强于 GCN，SAGE 蒸馏结果仍在运行。

单 seed 结果；TRD 目前与 Logits KD 接近，多数教师下低于 RKD-distance/angle。最终判断必须等 3 seeds 和 SAGE 学生完成。

### TRD 消融（GCN + TGN + seed42，第二块 GPU）

| 变体 | 官方 Test MRR |
|---|---:|
| temporal（主矩阵） | 0.4596 |
| static | 0.4608 |
| shuffled | 0.4470 |
| reverse | 0.4669 |
| zero / kd0 / rel0 | 运行/排队中 |

`shuffled` 破坏配对关系后下降约 0.013，说明关系增量中的查询配对对应有实际作用；`reverse` 在单 seed 下反而高于 temporal（0.4669 vs 0.4596），因此当前不能声称双前缀时序增量优于静态或反向控制。需要 3 seeds 和更多教师后再判断。

### 并行消融任务（第二块 GPU）

- `results/ablation-trd`：TRD 消融 13 runs（GCN/SAGE × static/shuffled/reverse/zero/kd0/rel0，SAGE 额外含 temporal）。复用 seed42 TGN 教师，不修改源码。
- `results/ablation-capacity`：学生容量消融 6 runs（hidden 16/64、layers 1/3、K 4/16），在 TRD 消融队列结束后自动启动。
- 两个队列均写入独立目录，不影响 `results/main` 的主矩阵和 source hash。

预计 Wiki 141 个 run 顺序执行约 4–7 天（RKD 的距离/角度目标在真实数据上显著更慢）。脚本 `finalize_wiki.sh` 会在矩阵结束后自动生成 `results/main/summary.csv` 和 `results/WIKI_DONE.txt`。

当前快照：`results/main/summary-partial.csv`（只包含已完成的 run，所有分组标记为 incomplete）。

## 重要约束

- 正式矩阵运行期间不要修改 `train.py`、`benchmark.py`、`model/`、`loss/`、`dataset/`、`util/`、`experiments/` 中的源码，否则 `source_hash` 变化会导致前后结果不可合并。
- 所有方法共享同一教师 checkpoint、同一学生初始化种子、同一数据切分和候选采样协议。
- 未虚构任何未运行的实验数据；上表只记录实际完成的 run。
