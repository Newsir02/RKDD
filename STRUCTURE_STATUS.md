# 结构交互蒸馏实验运行状态

- 启动时间：2026-11-15 16:25:40（服务器本地时间）
- 当前快照：2026-11-15 17:57（服务器本地时间）
- 当前状态：运行中，无 failure
- 代码版本指纹：`1208b42886b3badc13e8ffd8b220590cb9ba95fa01cb356541e7235954a23076`

## 1. 本轮范围

本轮按 `comparison_experiments.md` 的新方法 `structure_interaction` 重新验证，先完成 **Wiki v2**。

| 任务组 | 计划 | 说明 |
|---|---:|---|
| TGN→GCN 主对比 | 27 | 3 seeds ×（1 教师 + 1 监督 + 7 蒸馏） |
| 不同教师补充 | 12 | 3 seeds ×（TGAT、GraphMixer 教师 + 各自结构蒸馏） |
| 零目标消融 | 3 | 3 seeds × `structure-mode zero` |
| **Wiki 合计** | **42** | 只运行 Wiki；Review/Coin/Comment 尚未启动 |

Review、Coin、Comment 的全量训练在当前算力下不可行（Review 单 epoch 曾需 6308 秒），因此没有启动，也没有用旧结果或合成结果代填。

## 2. 验证记录

| 检查 | 结果 |
|---|---|
| 完整回归测试 | 147 passed |
| 合成结构矩阵 | 14/14 runs complete，0 failure |
| Wiki 1 epoch 真实训练 | eligible 9.86%，distilled 2.97%，官方 val MRR 0.4504 |
| 旧教师 checkpoint | 数据指纹不同，未复用；正式矩阵重新训练教师 |

## 3. 当前进度

总体：**5 / 42** 个 run 完成，0 failure。

| run | 状态 |
|---|---|
| `seed-42/teacher-tgn` | 完成 |
| `seed-42/gcn/supervised` | 完成 |
| `seed-42/gcn/tgn/logits` | 完成 |
| `seed-42/gcn/tgn/trd` | 运行中 |
| `seed-44/teacher-tgn` | 完成 |
| `seed-44/gcn/supervised` | 完成 |
| `seed-44/gcn/tgn/logits` | 运行中 |
| 其余 | 排队 |

当前结果：

| 数据集 | 教师 | 学生 | 方法 | seed | 官方 test MRR | 官方 val MRR | best epoch / epochs | 训练秒 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| tgbl-wiki | TGN | — | teacher | 42 | 0.6035 | 0.6418 | 28 / 48 | 1816.9 |
| tgbl-wiki | TGN | — | teacher | 44 | 0.6125 | 0.6424 | 20 / 40 | 1923.1 |
| tgbl-wiki | — | GCN | supervised | 42 | 0.4359 | 0.4967 | 8 / 28 | 307.6 |
| tgbl-wiki | — | GCN | supervised | 44 | 0.4408 | 0.5231 | 20 / 40 | 582.5 |
| tgbl-wiki | TGN | GCN | logits | 42 | 0.4621 | 0.5239 | 23 / 43 | 850.8 |

## 4. 运行进程

| 进程 | PID | 配置 | GPU | 计划 |
|---|---:|---|---:|---:|
| matrix A | 2298079 | `configs/structure_wiki_seeds42_43.json` | 0 | 28 runs |
| matrix B | 2298080 | `configs/structure_wiki_seed44.json` | 1 | 14 runs |

两个进程共用输出根 `results/structure-main`，但 run 目录按 seed 分开，不重叠。两个进程都会写 `results/structure-main/manifest.json`，该文件不是合并清单；每个 run 的 `job.json`、`config.json`、`result.json` 才是权威记录。日志为：

- `results/structure-main-a.log`
- `results/structure-main-b.log`

## 5. 查看与汇总

```bash
cd /data/ljh/RKD
bash structure_status.sh

PYTHONPATH=/data/ljh/RKD/.benchmark-deps \
  /home/ubuntu/anaconda3/envs/Graph/bin/python -m experiments.summarize \
  results/structure-main --output results/structure-main/summary.csv
```

## 6. 停止与恢复

停止：

```bash
kill "$(cat results/structure-main-a.pid)" "$(cat results/structure-main-b.pid)"
```

恢复（先移动未完成的 run 目录，否则 `benchmark` 会拒绝非空输出目录）：

```bash
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
nohup env CUDA_VISIBLE_DEVICES=0 /home/ubuntu/anaconda3/envs/Graph/bin/python \
  -m experiments.matrix --config configs/structure_wiki_seeds42_43.json \
  --output-root results/structure-main --execute --resume \
  > results/structure-main-a-resume.log 2>&1 &
echo $! > results/structure-main-a.pid

nohup env CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/Graph/bin/python \
  -m experiments.matrix --config configs/structure_wiki_seed44.json \
  --output-root results/structure-main --execute --resume \
  > results/structure-main-b-resume.log 2>&1 &
echo $! > results/structure-main-b.pid
```

`--resume` 会校验配置、源码指纹、checkpoint 和共享教师哈希；中断的 run 因不支持 epoch 级断点，会从第 1 epoch 重跑。

## 7. 产物

| 内容 | 路径 |
|---|---|
| 主矩阵 / 多教师 / 消融 | `results/structure-main/` |
| 正式任务配置 | `configs/structure_comparison.json` |
| Wiki 分卡配置 | `configs/structure_wiki_seeds42_43.json`、`configs/structure_wiki_seed44.json` |
| 合成验证 | `results/structure-smoke-20261115/` |
| 1 epoch 调试 | `results/structure-wiki-debug-teacher-1ep/`、`results/structure-wiki-debug-student-1ep/` |
| 状态脚本 | `structure_status.sh` |
