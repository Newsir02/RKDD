# 结构交互蒸馏 Wiki 实验状态

- 启动：2026-11-15 16:25:40（服务器时间）
- 完成：2026-11-16（服务器时间）
- 状态：**42 / 42 runs 完成，0 failure**
- 代码版本指纹：`1208b42886b3badc13e8ffd8b220590cb9ba95fa01cb356541e7235954a23076`
- 数据指纹：`15fdb2d49b464e85ef08881639cf9e644af9b6c527576159b9c01c11c82110df`

## 1. 范围

本轮只运行 **Wiki v2**，按 `comparison_experiments.md` 的结构交互方案生成 42 个 run：

| 任务组 | 数量 | 状态 |
|---|---:|---|
| TGN→GCN 主对比 | 27 | 完成 |
| TGAT / GraphMixer 不同教师补充 | 12 | 完成 |
| 零目标消融 | 3 | 完成 |
| **合计** | **42** | **完成** |

Review / Coin / Comment 未启动；这些数据集的全量训练在当前 2×4090 上不可行。

## 2. 验证记录

| 检查 | 结果 |
|---|---|
| 完整回归测试 | 147 passed |
| 合成结构矩阵 | 14 / 14 complete，0 failure |
| Wiki 1 epoch 调试 | eligible 9.86%，distilled 2.97%，官方 val MRR 0.4504 |
| 正式矩阵 failure | 0 |

旧 Wiki 教师 checkpoint 因数据指纹不同未复用；正式矩阵重新训练了三个教师。

## 3. Wiki 最终汇总

三个 seed（42/43/44）的官方 test MRR，均值 ± 标准差：

| 方法 | 教师 | Test MRR |
|---|---|---:|
| Logits KD | TGN | **0.4560 ± 0.0055** |
| 原 TRD | TGN | 0.4532 ± 0.0073 |
| **结构交互 / 零目标** | TGN | 0.4518 ± 0.0120 |
| **结构交互 / 真实目标** | TGN | 0.4424 ± 0.0055 |
| LightST 组件 | TGN | 0.4397 ± 0.0074 |
| 监督 GCN | — | 0.4384 ± 0.0025 |
| LKD4DyTAG 组件 | TGN | 0.4291 ± 0.0046 |
| DynGKD 组件 | TGN | 0.4274 ± 0.0087 |
| Distill2Vec 组件 | TGN | 0.4258 ± 0.0148 |

不同教师下的结构交互：

| 教师 | 教师 Test MRR | 结构交互学生 Test MRR |
|---|---:|---:|
| TGN | 0.6024 ± 0.0107 | 0.4424 ± 0.0055 |
| TGAT | 0.6403 ± 0.0047 | 0.4342 ± 0.0177 |
| GraphMixer | 0.6560 ± 0.0082 | 0.4414 ± 0.0102 |

## 4. 当前解释

- 真实结构交互目标 **0.4424** 略高于监督 GCN **0.4384**，但低于 Logits KD **0.4560** 和原 TRD **0.4532**。
- 同一套四视图、采样和 Huber 预算下，**零目标 0.4518 高于真实教师交互目标 0.4424**。
- 因此当前 Wiki 结果不足以支持“结构交互项带来独立增益”。在扩大数据集或增加方法之前，应先检查教师交互目标的能量/符号、候选对选择、遮蔽覆盖和超参数，而不是把 0.4424 解释为方法有效。

## 5. 产物

| 内容 | 路径 |
|---|---|
| 全部结果 | `results/structure-main/` |
| 最终汇总 | `results/structure-main/summary.csv` |
| 主配置 | `configs/structure_comparison.json` |
| Wiki 分卡配置 | `configs/structure_wiki_seeds42_43.json`、`configs/structure_wiki_seed44.json` |
| 合成验证 | `results/structure-smoke-20261115/` |
| 1 epoch 调试 | `results/structure-wiki-debug-teacher-1ep/`、`results/structure-wiki-debug-student-1ep/` |

复现汇总：

```bash
cd /data/ljh/RKD
PYTHONPATH=/data/ljh/RKD/.benchmark-deps \
  /home/ubuntu/anaconda3/envs/Graph/bin/python -m experiments.summarize \
  results/structure-main --output results/structure-main/summary.csv
```
