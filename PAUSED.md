# 实验暂停记录

- 暂停时间：2026-11-15 14:55:39（服务器本地时间）
- 操作：已停止 Wiki 主矩阵、TRD 消融、容量消融等待器、Wiki 自动收尾等待器
- 当前没有实验进程在运行；GPU 上剩余显存属于其他用户进程

## 已完成进度

| 队列 | 完成 | 总数 | 最后完成的 run |
|---|---:|---:|---|
| Wiki 主矩阵 | 31 | 141 | `sage/tgn/logit_standard` |
| TRD 消融 | 8 | 13 | `sage/trd-static` |
| 学生容量消融 | 0 | 6 | 尚未启动 |

- `failure.json`：主矩阵 0，消融 0
- `WIKI_DONE.txt` 和 `results/main/summary.csv` 尚未生成（主矩阵未结束）
- 自动收尾进程已停止，不会在后台继续写入

## 被中断的 run

以下两个 run 在停止时尚未完成，已移到 `results/incomplete/20261115-145539/`，保留 config 和 epochs 日志：

- `results/incomplete/20261115-145539/main/tgbl-wiki/seed-42/sage/tgn/nosmog_rsd/`
- `results/incomplete/20261115-145539/ablation-trd/tgbl-wiki/seed-42/sage/trd-shuffled/`

`benchmark` 和消融 runner 不支持 epoch 级断点，因此恢复时这两个 run 会从第 1 epoch 重新开始；已完成的结果不受影响。

## 恢复命令

### Wiki 主矩阵

已完成 31 个，剩余 110 个：

```bash
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export CUDA_VISIBLE_DEVICES=1
nohup /home/ubuntu/anaconda3/envs/Graph/bin/python -m experiments.matrix \
  --config configs/comparison.json \
  --output-root results/main \
  --execute --max-runs 110 --resume \
  > results/main-wiki-resume.log 2>&1 &
echo $! > results/main-wiki.pid
```

注意：这里必须用 `--max-runs 110`（当前剩余数）。如果恢复时仍写 `--max-runs 141`，`--resume` 跳过的 31 个不计入 `completed`，主矩阵在跑完 Wiki 后可能继续进入 Review 任务。

### TRD 消融

已完成 8 个，剩余 5 个（SAGE 的 shuffled/reverse/zero/kd0/rel0）：

```bash
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export CUDA_VISIBLE_DEVICES=0
nohup /home/ubuntu/anaconda3/envs/Graph/bin/python run_trd_ablation.py \
  > results/ablation-trd-resume.log 2>&1 &
echo $! > results/ablation-trd.pid
```

### 容量消融

TRD 消融结束后再启动：

```bash
cd /data/ljh/RKD
export PYTHONPATH=/data/ljh/RKD/.benchmark-deps
export CUDA_VISIBLE_DEVICES=0
nohup /home/ubuntu/anaconda3/envs/Graph/bin/python run_capacity_ablation.py \
  > results/ablation-capacity.log 2>&1 &
echo $! > results/ablation-capacity.pid
```

### 自动收尾（可选）

Wiki 主矩阵结束后重新启动 `finalize_wiki.sh` 即可生成最终 `summary.csv` 和 `WIKI_DONE.txt`。
