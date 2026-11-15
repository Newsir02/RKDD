# Wiki v2 实验诊断与 TRD 改进

2026-11-15。正式结果来自 [EXPERIMENT_RESULTS.md](../../EXPERIMENT_RESULTS.md) 的 Wiki v2、seed 42 快照。该快照引用的服务器检查点和逐轮日志当前不在本地工作区。下文的本地训练使用重新下载的 Wiki v2 事件数据，结果单列。

## 正式结果能说明什么

| GCN 的教师 | Logits KD | 原 TRD | TRD − Logits KD |
|---|---:|---:|---:|
| TGN | 0.4600 | 0.4596 | −0.0004 |
| TGAT | 0.4680 | 0.4637 | −0.0043 |
| GraphMixer | 0.4726 | 0.4632 | −0.0094 |
| 三个教师简单平均 | 0.4669 | 0.4622 | −0.0047 |

原 TRD 在三个教师下都高于独立监督 GCN（0.4373），但加上余弦增量后，没有超过匹配教师的 Logits KD。差距最大的是 GraphMixer。平均训练时间从 869.9 秒变为 1223.4 秒，约增加 41%，同时包含实际早停轮数的影响。三个教师共用同一个种子，不能当作三个独立随机重复。

TGN→GCN 的 `static`、`reverse`、`zero` 分别得到 0.4608、0.4669、0.4674，均高于 `temporal` 的 0.4596。`shuffled` 得到 0.4470，说明配对破坏会改变结果，但不足以证明正确符号有增益。当前数据与“增量项主要提供平滑正则，而教师指定的方向不一定有益”相容。SAGE 版原 TRD 在该快照中尚无结果。

原 TRD 对齐源节点与候选的余弦变化。实际排序由可学习的边解码器完成，解码器没有约束为余弦分数的单调函数。历史更新也可能对教师和学生产生不同幅度的改动，因为两者的历史邻居预算和状态更新机制不同。单看 `rkd` 数值无法确定对排序梯度的影响，训练日志应同时看教师/学生增量能量、与实际解码分数变化的一致性，以及与 `zero` 对照的配对差值。

## 为什么 SAGE 会超过 TGN

独立监督 SAGE 的 0.6535 已超过 TGN 的 0.6114，领先 0.0421。对 TGN 做 Logits KD 后，SAGE 再上升 0.0162 到 0.6697。监督 SAGE 同时略高于 TGAT（0.6450），低于 GraphMixer（0.6654）。这些差值说明，学生胜过某个教师不能直接归因于蒸馏，教师的总体 MRR 也不是学生的上限。

- **身份和容量。** Wiki 的 `9227 × 32 = 295264` 个可训练节点 ID 参数单独就超过 TGN 全部 134945 个参数。GCN 也使用相同的 ID 表，SAGE 还在每层用独立的根节点分支保留自身信息。当前 GCN 会把根节点与邻居一起按度数混合，频繁交互节点的身份信息可能被稀释。这个机制与差距一致，贡献大小需要干预实验。
- **重复关系。** 本地完整事件表中，60.62% 的测试事件，其 `(src,dst)` 在训练集已出现。节点 ID、最近历史快照和监督解码器都可利用这种规律。这项统计不表示 SAGE 的每次正确预测都来自记忆。关闭 `--student-node-ids` 可以检验身份表依赖，在无节点属性的 Wiki 上也会同时移除学生区分节点的重要信息。
- **训练和候选。** 三个教师的最佳轮分别为 22、2、2，监督 SAGE 为 1。共享预算和验证规则允许不同最佳轮，不能用最佳轮长短判断充分训练。训练的 16 候选与正式评估近 1,000 候选存在目标差异。要判断外推性，应按训练中是否见过节点对分层报告、固定相同候选，按验证结果比较更多种子。

## 排行榜候选审计

本地下载的数据指纹为 `15fdb2d49b464e85ef08881639cf9e644af9b6c527576159b9c01c11c82110df`，与原快照前缀一致。边表中有 8,227 个源节点和 1,000 个目的页面。逐条比较官方文件与完整目的域后，验证、测试各 23,621 个查询均覆盖除同一 `(src,t)` 的所有正页面以外的完整目的域。每个切分各有 23,619 条带 999 个负例，2 条带 998 个负例。因而“每条约 999 个负例”与官方 all-negative 描述一致。文件哈希和核对计数见 `results/wiki-negative-audit.json`。[TGB 官方排行榜的 Wiki v2 协议](https://tgb.complexdatalab.com/docs/leader_linkprop/#leaderboard-for-tgbl-wiki-v2)

原快照没有保存负例文件哈希供本地对比。其榜单差距的归因仍应核对教师实现、打分前的历史更新批次、同时间戳事件可见性、以及训练与模型选择。`official=true` 记录调用的评估器和负例来源，这些信息不足以证明整个训练、推理流程与榜单实现一致。

## 候选分数增量

新增 `trd_score` 继续对同一查询、同一候选和同一模型权重使用两个因果前缀。设 `z⁺, z⁻` 分别为加入上一批真实交互之后、之前的 logits，`Δ = ((z⁺−z⁻)−mean_c(z⁺−z⁻))/T`。任意两个分量的差是相应候选的温度化 log-odds 变化。中心化消去所有候选的共同偏移，SmoothL1 对大误差采用线性惩罚。损失为 `CE + λKD·KL + λΔ·SmoothL1(Δs, Δt)`，`--score-delta-warmup` 控制增量项系数的线性爬升。

`--trd-mode zero/reverse/shuffled` 复用同样的前向与损失权重，只替换教师目标，检验平滑、符号和查询配对。`--relation-weight 0` 自动跳过第二个历史前缀。训练日志记录双方分数增量能量、教师对正例的相对分数变化、教师余弦增量与分数增量的逐查询平均相关性。该方法不改变学生推理结构。

`--score-delta-gate` 采用可选的逐查询教师门控。计算师生在历史更新后对训练正例的 CE。教师 CE 更低时，该查询才计入 KD 和增量损失。门控本身不求导，两个蒸馏项按整批平均，监督 CE 始终覆盖整批。`--relation-weight 0 --score-delta-gate` 是配对的门控 Logits KD，用来判断收益来自教师筛选还是方向增量。门控使用训练候选上的正例概率作为可靠性代理，正式全候选排名仍要单独验证。

## 三种子小规模实测

在 RTX 4060 Laptop GPU 上完成 66 次训练。使用 Wiki v2 的前 4,000 条真实事件重新按 70/15/15 切分，训练和评估每查询随机 16 个候选。种子为 42/43/44，TGN 训练 3 轮，学生 4 轮，按验证 MRR 选权重。同一 seed 的蒸馏均复用一个 TGN checkpoint，已核对文件哈希。该试验的 `official=false`，下表与正式 Wiki v2 的全量 MRR 独立解释。

| 设置 | GCN test MRR | SAGE test MRR |
|---|---:|---:|
| 监督学生 | **0.6287 ± 0.0086** | 0.6101 ± 0.0177 |
| 监督学生，关闭节点 ID | — | 0.3104 ± 0.0308 |
| Logits KD | 0.6096 ± 0.0129 | 0.6073 ± 0.0234 |
| 原 TRD | 0.6131 ± 0.0134 | 0.6262 ± 0.0310 |
| 候选分数增量 | 0.6096 ± 0.0187 | 0.6253 ± 0.0188 |
| 分数增量目标置零 | 0.6159 ± 0.0220 | 0.6272 ± 0.0288 |
| 分数增量目标反向 | 0.6154 ± 0.0233 | 0.6298 ± 0.0195 |
| 分数增量目标打乱 | 0.6167 ± 0.0260 | 0.6095 ± 0.0335 |
| 门控 Logits KD | 0.6244 ± 0.0168 | 0.6070 ± 0.0245 |
| 门控 + 分数增量 | 0.6243 ± 0.0162 | 0.6109 ± 0.0237 |
| 门控 + 分数增量目标置零 | 0.6257 ± 0.0195 | 0.6098 ± 0.0223 |

该预算下的 TGN 为 `0.4436 ± 0.0577`。关闭节点 ID 后 SAGE 的参数从 284,097 降到 8,929，MRR 约下降 0.300。此消融强烈支持身份表在该设定中的作用。移除 ID 模块也改变了后续层的随机初始化序列，且短训练中有 ID 的 GCN 高于 SAGE，因此不能从该试验估计 SAGE 根分支对全量数据差距的独立贡献。

原 TRD 在最后一轮的训练日志中，教师余弦增量平方均值约 0.149，GCN 约 0.00017，SAGE 约 0.058。教师余弦增量与解码分数增量的逐查询中心化余弦一致性约 0.35。前者显示 GCN 的增量可表达性与教师差异大，后者显示余弦方向只部分对应解码器的变化。分数目标更直接，但同样没有稳定超过它的 `zero`、`reverse`，所以目前仍没有证据表明正确方向带来增益。

门控使 GCN 的短训结果相对普通 Logits KD 提高约 0.0148，仍低于独立监督。SAGE 的门控 Logits KD 与普通 Logits KD 接近，门控加分数增量的验证 MRR 为 0.6598，低于门控 Logits KD 的 0.6697。最后一轮教师门控通过率在 GCN 上约 7%，在 SAGE 上约 13%。这个短训教师与服务器正式教师强度不同，门控的收益范围仍需复核。

## 复现与后续正式对照

```bash
python -m experiments.matrix --config configs/wiki_trd_pilot.json \
  --output-root results/wiki-trd-reproduction --execute
python -m experiments.summarize results/wiki-trd-reproduction \
  --output results/wiki-trd-reproduction/summary.csv
```

对同一个 `trd_score/job.json` 改 `--trd-mode zero`、`reverse`、`shuffled` 可以复现方向消融。对 `sage/supervised/job.json` 加 `--no-student-node-ids` 复现身份消融。以下命令复用正式 `Logits KD` 任务的配置和同一个教师文件，在新目录中运行三组配对试验。

```bash
job=results/main/tgbl-wiki/seed-42/sage/tgn/logits/job.json
python benchmark.py --config "$job" --method trd_score --score-delta-warmup 3 \
  --output-dir results/wiki-followup/seed-42/sage/tgn/score
python benchmark.py --config "$job" --method trd_score --score-delta-gate --relation-weight 0 \
  --output-dir results/wiki-followup/seed-42/sage/tgn/gated-logits
python benchmark.py --config "$job" --method trd_score --score-delta-gate --score-delta-warmup 3 \
  --output-dir results/wiki-followup/seed-42/sage/tgn/gated-score
```

把 `--trd-mode zero` 加到对应的 score 配置，得到同训练预算的方向对照。正式复核可先固定 TGN→GCN 和 TGN→SAGE，用 seeds 42/43/44，随后用表现最强的 GraphMixer 验证结论是否依赖教师。验证集先比较 `λΔ=0, 0.25, 1`，并按同种子报告 `temporal−zero` 和 `score−logits` 的差值。原服务器批量任务继续使用原代码与结果目录，新版本的源码指纹会使原目录的 `--resume` 拒绝跳过旧记录。单次新蒸馏仍可复用通过教师契约检查的旧 checkpoint。

本地逐种子数值和原始产物路径见 `results/wiki-trd-paired-results.csv`。静态编译通过，完整回归为 130 项通过。原余弦 TRD 损失与双前缀训练器的 SHA256 与修改前相同。全量官方 MRR 上的新方法增益尚无实测。
