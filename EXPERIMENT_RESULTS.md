# TRD 实验数据汇总

- 快照时间：2026-11-15
- 当前状态：Wiki 主矩阵运行中（已完成 29 / 141 个 run，当前 `SAGE / TGN / RKD-angle`）
- 评价指标：TGB 官方 test MRR（`official=true`）
- 当前正式结果只有 **Wiki v2、seed42**；Review 只有 1-epoch 教师调试；Coin / Comment / Flight 未进行模型训练。
- 所有数字均来自实际 `result.json`，未虚构未运行的实验。

后续补充见 [Wiki v2 实验诊断与 TRD 改进](docs/research/trd_wiki_analysis.md)。本报告保留服务器快照的正式数值，本地小规模结果在补充文档单列。

---

## 1. 环境与验证

| 项目 | 内容 |
|---|---|
| Python | 3.10.19（conda `Graph` 环境） |
| PyTorch | 2.1.0+cu118 |
| PyG | 2.6.1 |
| py-tgb | 2.2.0（通过 `PYTHONPATH=.benchmark-deps`） |
| GPU | 2 × RTX 4090 |
| 单元测试 | 126 passed |
| 合成 smoke | 141 / 141 runs 成功，0 failure，47 个 3-seed 分组完成 |

合成 smoke 只验证执行链，tiny synthetic 分数不作为方法性能结论。

---

## 2. 数据集

五个 TGB 数据集均已完成下载、zip 尺寸/CRC 校验、官方切分和负例文件校验。数据不进入 Git。

| 数据集 | 节点 | 事件 | train | val | test | data signature（前 8 位） |
|---|---:|---:|---:|---:|---:|---|
| tgbl-wiki v2 | 9,227 | 157,474 | 110,232 | 23,621 | 23,621 | `15fdb2d4` |
| tgbl-review v2 | 352,637 | 4,873,540 | 3,413,837 | 730,784 | 728,919 | `20ab4e59` |
| tgbl-coin v2 | 638,486 | 22,809,486 | 15,966,659 | 3,421,416 | 3,421,411 | `a59d50be` |
| tgbl-comment v1 | 994,790 | 44,314,507 | 31,020,155 | 6,647,176 | 6,647,176 | `1bdc0013` |
| tgbl-flight v2 | 18,143 | 67,169,570 | 47,043,612 | 10,099,015 | 10,026,943 | `511cc351` |

数据清单：`results/{wiki,review,coin,comment,flight}-data.json`  
数据哈希：`results/data-sha256.txt`

---

## 3. Wiki v2 教师结果（seed42）

| 教师 | 官方 val MRR | 官方 test MRR | best epoch / epochs | 训练秒 | 参数量 |
|---|---:|---:|---:|---:|---:|
| TGN | 0.6385 | 0.6114 | 22 / 42 | 1539.4 | 134,945 |
| TGAT | 0.6898 | 0.6450 | 2 / 22 | 423.4 | 707,905 |
| GraphMixer | 0.6972 | **0.6654** | 2 / 22 | 360.0 | 690,881 |

Wiki 上 **GraphMixer > TGAT > TGN**。三个教师均通过契约、数据指纹和 checkpoint SHA256 校验，并被所有学生/蒸馏方法共享。

---

## 4. Wiki v2 GCN 学生结果（seed42）

### 4.1 按教师

| 方法 | TGN | TGAT | GraphMixer |
|---|---:|---:|---:|
| 监督 GCN | 0.4373 | — | — |
| Logits KD | 0.4600 | 0.4680 | **0.4726** |
| TRD | 0.4596 | 0.4637 | 0.4632 |
| RKD-distance | 0.4700 | **0.4725** | 0.4678 |
| RKD-angle | 0.4646 | 0.4709 | 0.4725 |
| DKD | 0.4473 | 0.4445 | 0.4701 |
| Logit Standardization | 0.4485 | 0.4488 | 0.4490 |
| NOSMOG-RSD component | 0.2293 | 0.4189 | 0.4433 |

### 4.2 方法均值与训练成本

下表为三个教师上的简单均值（不是 3-seed 均值）。

| 方法 | Test MRR 均值 | 平均训练秒 |
|---|---:|---:|
| RKD-distance | **0.4701** | 8175.1 |
| RKD-angle | 0.4693 | 4224.3 |
| Logits KD | 0.4669 | 869.9 |
| TRD | 0.4622 | 1223.4 |
| DKD | 0.4540 | 535.7 |
| Logit Standardization | 0.4488 | 502.1 |
| 监督 GCN | 0.4373（仅 TGN） | 273.9 |
| NOSMOG-RSD component | 0.3639 | 878.9 |

GCN 上当前较强的是 **RKD-distance / RKD-angle / Logits KD**；TRD 与 Logits KD 接近，暂未表现出优势。RKD-distance 明显更慢。NOSMOG-RSD 的未归一化 Gram 目标不稳定，TGN 教师下只有 0.2293。

---

## 5. Wiki v2 SAGE 学生结果（进行中）

| 方法 | 教师 | Test MRR | best epoch / epochs | 训练秒 |
|---|---|---:|---:|---:|
| 监督 SAGE | — | 0.6535 | 1 / 21 | 227.4 |
| Logits KD | TGN | **0.6697** | 3 / 23 | 455.8 |
| DKD | TGN | 0.6661 | 2 / 22 | 503.4 |
| RKD-distance | TGN | 0.6571 | 2 / 22 | 3712.9 |
| TRD / RKD-angle / 其他 | TGN | 运行中 | — | — |

SAGE 明显强于 GCN：监督学生 **0.6535 vs 0.4373**；SAGE + Logits KD 已达 0.6697。SAGE 的完整 7 方法结果尚未跑完。

---

## 6. TRD 消融（GCN + TGN + seed42）

| 变体 | Test MRR | best epoch / epochs | 训练秒 | 状态 |
|---|---:|---:|---:|---|
| temporal（主矩阵） | 0.4596 | 14 / 34 | 1032.4 | 完成 |
| static | 0.4608 | 7 / 27 | 1027.1 | 完成 |
| shuffled | 0.4470 | 14 / 34 | 1512.0 | 完成 |
| reverse | 0.4669 | 9 / 29 | 1094.0 | 完成 |
| zero | 0.4674 | 17 / 37 | 1248.8 | 完成 |
| kd0（去掉 logits KD） | — | — | — | 运行中 |
| rel0（去掉关系项） | — | — | — | 排队 |

初步观察：

- `shuffled` 破坏配对关系后下降约 0.013（0.4470 vs 0.4596），说明查询配对对应有实际作用。
- `static` 与 `temporal` 接近；`reverse` 和 `zero` 在单 seed 下反而更高。
- 因此当前不能声称双前缀时序增量优于静态/反向/零控制项，必须等 3 seeds 和更多教师。

---

## 7. 学生容量消融

已配置 6 个 run，等待 TRD 消融结束后自动启动：

- hidden：16 / 64（主配置 32）
- layers：1 / 3（主配置 2）
- history K：4 / 16（主配置 8）

当前尚无结果。

---

## 8. Review v2 真实规模调试

只完成 1-epoch TGN 教师调试，不是正式教师：

| 项目 | 数值 |
|---|---:|
| 节点 / 事件 | 352,637 / 4,873,540 |
| 官方 val MRR | 0.2979 |
| 官方 test MRR | 0.3782 |
| 1 epoch 训练时间 | 6308.2 秒（约 1.75 小时） |
| 峰值显存 | 约 1.69 GB |
| 参数量 | 80,225 |

Review 没有学生蒸馏结果；单 epoch 已需 1.75 小时，全量矩阵在当前算力下不可行。

---

## 9. Coin / Comment / Flight

只完成数据准备，没有模型训练结果：

| 数据集 | 节点 | 事件 | 状态 |
|---|---:|---:|---|
| Coin v2 | 638,486 | 22,809,486 | 数据已校验，未训练 |
| Comment v1 | 994,790 | 44,314,507 | 数据已校验，未训练 |
| Flight v2 | 18,143 | 67,169,570 | 数据已校验，未训练 |

事件量分别是 Wiki 的 145 / 281 / 427 倍；按 Review 的实测速度，正式矩阵需要数月。没有虚构任何性能数字。

---

## 10. Wiki 1-epoch 调试结果

仅用于接口验证，不是正式性能：

| 模型/方法 | 1-epoch Test MRR |
|---|---:|
| TGN 教师 | 0.3267 |
| 监督 GCN | 0.3845 |
| Logits KD GCN | 0.3827 |
| DKD GCN | 0.3956 |
| RKD-distance GCN | 0.3802 |
| RKD-angle GCN | 0.4110 |
| Logit Standardization GCN | 0.3723 |
| NOSMOG-RSD GCN | 0.0584 |
| TRD GCN | 0.3660 |
| TRD SAGE | 0.6460 |

这些数字不能与正式 100-epoch 结果混合比较。

---

## 11. 与官方排行榜的关系

TGB 官方 `tgbl-wiki-v2` 排行榜使用 **all possible negative destinations** 协议（package ≥0.7.5），官方同名模型：

| 教师 | 官方 v2 榜单 Test MRR | 本项目 seed42 |
|---|---:|---:|
| TGN | 0.396 ± 0.060 | 0.6114 |
| TGAT | 0.141 ± 0.007 | 0.6450 |
| GraphMixer | 0.118 ± 0.002 | 0.6654 |

**候选协议补充核对。** Wiki v2 有 1,000 个目的页面。本地重新下载的官方 `val_ns_v2.pkl` 和 `test_ns_v2.pkl` 中，每个切分的 23,619 个查询有 999 个负例，另外 2 个查询有 998 个负例。逐条核验均覆盖完整目的域，排除了同一源节点和时间戳的所有正例。负例数与排行榜的 all-negative 描述一致。服务器快照的负例文件哈希还有待核对。[TGB 官方 Wiki v2 评估协议](https://tgb.complexdatalab.com/docs/leader_linkprop/#leaderboard-for-tgbl-wiki-v2)

此外：

- 教师是 project implementations，不是官方骨干；
- 只有 seed42，没有多 seed 均值±标准差；
- 未完成原论文排行榜复现。

当前数值与榜单同名模型之间的差距还需要结合具体骨干实现、历史更新批次和模型选择流程解释。单凭负例数无法判定差距由候选协议引起。

---

## 12. 结论与未完成项

### 当前可下的结论

- Wiki 教师可用；GraphMixer 最强（0.6654），TGAT 次之（0.6450），TGN 最低（0.6114）。
- Wiki SAGE 明显强于 GCN：监督 0.6535 vs 0.4373。
- GCN 上 RKD-distance/angle 与 Logits KD 较强，TRD 与 Logits KD 接近。
- TRD 消融中 `shuffled` 下降，但 `reverse`/`zero` 在单 seed 下不低于 temporal，暂不能证明时序增量的优势。
- NOSMOG-RSD 组件在当前未归一化 Gram 目标下不稳定。

### 未完成

- Wiki 主矩阵剩余 run（SAGE 其余方法、seeds 43/44）。
- 3-seed 均值±标准差和最终 `summary.csv`。
- TRD 消融的 kd0/rel0 和 SAGE 部分。
- 学生容量消融。
- Review / Coin / Comment / Flight 的正式教师和学生实验；当前算力下不可行。
- 官方排行榜 all-negative 协议与官方骨干校验。

---

## 13. 产物位置

| 内容 | 路径 |
|---|---|
| 主矩阵结果 | `results/main/` |
| TRD 消融 | `results/ablation-trd/` |
| 容量消融 | `results/ablation-capacity/`（未启动） |
| Review 调试 | `results/review-debug/` |
| Wiki 1-epoch 调试 | `results/wiki-debug/` |
| 合成 smoke | `results/smoke-cuda-check/` |
| 全部结果汇总 JSON | `results/all-results.json` |
| 运行状态 | `results/STATUS.md` |
| 数据清单 | `results/*-data.json` |
| 数据哈希 | `results/data-sha256.txt` |

复现命令：

```bash
PYTHONPATH=.benchmark-deps python -m experiments.matrix \
  --config configs/comparison.json --output-root results/main --execute
PYTHONPATH=.benchmark-deps python -m experiments.summarize \
  results/main --output results/main/summary.csv
```
