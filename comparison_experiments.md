# 面向连续时间图链接预测的时序结构交互关系蒸馏：对比实验

## 1. 实验范围

主实验用 **TGN 教师、普通快照 GCN 学生**，在四个 TGB 数据集上比较蒸馏方法。不同教师实验固定新方法，分别比较 TGN、TGAT、GraphMixer 蒸馏同一 GCN 配置的结果。消融只在 Wiki v2 上做 **真实教师交互目标与零目标** 的一组配对比较。三个教师分别训练，各自蒸馏一个 GCN。主实验中的 TGN 新方法结果同时用于不同教师表和消融表。

| 数据集 | 数据版本 | 节点数 | 事件数 | 主指标 |
|---|---|---:|---:|---|
| `tgbl-wiki` | v2 | 9,227 | 157,474 | MRR |
| `tgbl-review` | v2 | 352,637 | 4,873,540 | MRR |
| `tgbl-coin` | v2 | 638,486 | 22,809,486 | MRR |
| `tgbl-comment` | v1 | 994,790 | 44,314,507 | MRR |

统计按 TGB 官方页面及 `py-tgb==2.2.0` 数据登记。现有本地小规模训练只使用 Wiki v2 前缀，其余三份完整数据需要在正式运行环境准备。数据集保留官方切分和验证、测试固定负例。Wiki v2 的候选域为约 1,000 个目的页面，999 个负例可以覆盖完整目的域中的其他页面。 [TGB 链接预测数据集](https://tgb.complexdatalab.com/docs/linkprop/) · [Wiki 候选审计](docs/research/trd_wiki_analysis.md#排行榜候选审计)

## 2. TGN→GCN 主对比

| 运行项 | CLI 方法 | 对照的监督信号 |
|---|---|---|
| 教师 | `--stage teacher` | TGN 监督参考 |
| 学生 | `supervised` | GCN 链接交叉熵 |
| Logits KD | `logits` | 完整视图候选概率分布 |
| 原时序关系增量 | `trd` | 两前缀之间的余弦关系增量 |
| Distill2Vec 组件 | `distill2vec_latent` | 末层节点表示分布 KL |
| DynGKD F2 组件 | `dyngkd_structure` | 结构层表示分布 KL |
| LightST 组件 | `lightst_adaptive` | 预测 KL 与空间加权对比 |
| LKD4DyTAG 组件 | `lkd4dytag_edge` | 边表示 Hadamard 乘积与负指数余弦对齐 |
| 本文方法 | `structure_interaction` | 四视图候选相对分数的非加性交互 |

论文对照覆盖真实动态图表示、时空预测及动态文本图三类原任务。由于本协议统一使用 TGN、GCN 和 TGB 链接预测，表中四项复现的是可迁移损失组件。Distill2Vec 与 DynGKD 属于较早的动态图蒸馏研究。LightST 原任务是交通预测，LKD4DyTAG 的文本教师是 LLM，出版状态为 AAAI 2025 AI4TS workshop。公开发表的原文、作者仓库、精确公式、实现差异及验证请参考 [四篇论文复现说明](docs/research/temporal_baselines.md)。 [Distill2Vec，ASONAM 2020](https://doi.org/10.1109/ASONAM49781.2020.9381315) · [DynGKD，2021](https://link.springer.com/article/10.1007/s13278-021-00816-1) · [LightST，AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32096) · [LKD4DyTAG，2025](https://arxiv.org/abs/2502.10914)

软件入口为 [benchmark.py](benchmark.py)。同一数据集和种子只训练一份 TGN，所有对应的蒸馏方法加载同一 checkpoint 并核对哈希。监督与蒸馏共享 GCN 骨干、初始模型种子、候选采样与优化预算。主表只用现有 TGN 教师，TGAT 和 GraphMixer 不运行论文对照、Logits KD 或原 TRD。

## 3. 结构交互蒸馏

在查询 $q=(u,\mathcal C,t)$ 的批前缀内，选训练目标 $v$ 和同一候选组的一项参照 $w$。在有界历史快照中寻找 $u-a-b-v$ 路径，固定中间历史边。证据组 $A$ 为 $(u,a)$ 的历史事件，$B$ 为 $(b,v)$ 的历史事件。两端只取师生显式邻居缓存中共有的事件 ID。默认每组至多保留最近 2 个共同事件，选中后整组开关，重复交互按各自事件 ID 处理。每个训练批在合法查询中按固定随机源最多抽取 4 个查询进行关系对齐。其他查询仍参与完整图上的链接 CE。

四个视图保留同一查询和候选，分别保留两组中的零组、仅 $A$、仅 $B$、两组。记 $R_M^{ij}=z_M(u,v\mid H_{ij})-z_M(u,w\mid H_{ij})$，则

$$
I_M=R_M^{11}-R_M^{10}-R_M^{01}+R_M^{00},\qquad
\mathcal{L}=\mathcal{L}_{\mathrm{link}}+
\lambda\operatorname{Huber}_{\delta}\big(I_S-\operatorname{sg}(I_T)\big).
$$

- **学生**从真实前缀维护有界快照。四个视图在锁定的邻居缓存中遮蔽事件，不补入更早邻居，并按视图重算 GCN 度归一化。
- **TGN** 在四视图间固定由真实前缀构成的循环记忆，并遮蔽查询图上对应的显式邻居消息。该处理测量**固定记忆条件下的显式读出交互**，未撤销历史事件对记忆的影响。
- **TGAT 与 GraphMixer** 从已锁定的邻居槽中遮蔽对应事件，不补位。三教师实验因此比较各骨干的显式历史邻居读出响应。仅中间边在学生快照中需构成可达路径，不能假定每个教师都显式编码了该三跳路径。
- 所有四视图查询均在写入当前真实边之前完成。各分支禁止更新教师或学生的主事件缓存。沿用批前缀，允许已处理的同时间戳跨批事件可见；此处未采用按严格时间分组的另一种协议。
- 推理只在未遮蔽快照上运行 GCN。教师和训练视图全部离线于部署。完整批候选 CE 与逐查询四视图分别计算，未将不同查询的遮蔽合成一个全局遮蔽。

以 `eligible_fraction` 和 `distilled_fraction` 区分有合法路径的比例及实际抽取进行关系监督的比例。记录平均组大小和师生交互能量。训练集整轮没有合法查询时任务报错，避免把纯监督训练计为新蒸馏方法。

参数为 `--structure-group-events`、`--structure-queries-per-batch`、`--structure-huber-beta`（Huber 截断 $\delta$）和 `--relation-weight`。设计说明及结构图见 [distillation_methods.md](distillation_methods.md)。

## 4. 不同教师与单项消融

| 实验 | 数据 | 学生 | 教师 | 设置 |
|---|---|---|---|---|
| 不同教师 | Wiki、Review、Coin、Comment | GCN | TGN、TGAT、GraphMixer 各自独立 | 只用 `structure_interaction` |
| 单项消融 | Wiki v2 | GCN | TGN | `structure-mode temporal` 对 `structure-mode zero` |

零目标消融保留同一个教师、完全相同的抽样规则、四种视图、四次教师打分和 Huber 系数，只把对齐目标从 $I_T$ 替换成 0。默认使用相同种子 42、43、44。TGN 的真实目标结果复用主实验。零目标对照检验教师的非零方向与幅度是否优于约束学生交互为零。若正文要进一步归因到剪除真实路径而非一般的多视图训练，还需另做等视图 Logits 或匹配随机路径检验。这两项不纳入目前约定的一组消融。

## 5. 预算和运行矩阵

默认配置见 [configs/structure_comparison.json](configs/structure_comparison.json)。全部使用 2 层 GCN、hidden 32、节点 ID 特征、最近 8 次历史交互。教师 hidden 64、24 个邻居槽。训练批大小 128，每条 16 个候选。AdamW 初始学习率 0.001，最多 100 轮，验证早停 patience 20。通用温度为 2，当前损失权重初值为 1。验证集用于选超参数和最佳轮次，所有方法保留相同的搜索预算。

| 任务组 | 总数 | 矩阵与复用规则 |
|---|---:|---|
| 主对比 | 108 | 4 数据集 × 3 种子 ×（1 教师 + 1 监督 GCN + 7 蒸馏） |
| 不同教师补充 | 48 | 4 数据集 × 3 种子 ×（2 个附加教师训练 + 2 个新方法蒸馏） |
| Wiki 消融补充 | 3 | 1 数据集 × 3 种子 × 1 个零目标运行 |
| **计划合计** | **159** | TGN 新方法和教师 checkpoint 在对应实验间复用 |

验证从训练历史重放，测试从训练与验证历史重放。官方全量任务使用原固定负例文件和评价器计算 MRR。结果还应报训练用时、GPU 峰值显存、参数量、前向延迟与历史状态开销。正式论文可按训练中节点对是否见过补充分层 MRR。本轮自动汇总只输出总体 MRR，分层表须后续按同一评估日志计算。大数据集完整 100 轮矩阵需要按设备与容量另行排期。

## 6. 复现命令与产物

安装依赖后，先验证合成训练链。矩阵命令缺省只写计划，指定 `--execute` 才训练。

```bash
python -m pip install -r requirements-benchmark.txt
python -m experiments.matrix --config configs/structure_smoke.json \
  --output-root results/structure-smoke --execute
python -m experiments.prepare_data \
  --datasets tgbl-wiki tgbl-review tgbl-coin tgbl-comment \
  --data-root datasets --download
python -m experiments.matrix --config configs/structure_comparison.json \
  --output-root results/structure-main
python -m experiments.matrix --config configs/structure_comparison.json \
  --output-root results/structure-main --execute
python -m experiments.summarize results/structure-main \
  --output results/structure-main/summary.csv
```

同样的参数可以传给 `scripts/run_comparison.sh`。需要跳过已有完整结果时，在相同源码、相同配置和未修改的 checkpoint 下用 `--execute --resume`。正式矩阵路径分为 `main/`、`multi-teacher/` 和 `ablation/`。同一目录保存 `job.json`、`epochs.jsonl`、模型、`result.json`。汇总表用 `structure_mode` 区分真实和零目标，只有包含全部配对种子的分组才标完整。

也可从已生成的主实验任务配置启动单独训练：

```bash
job=results/structure-main/main/tgbl-wiki/seed-42/gcn/tgn/structure_interaction/job.json
python benchmark.py --config "$job" --structure-mode zero \
  --output-dir results/wiki-standalone-zero
```

旧 `configs/comparison.json` 仍可复现原大矩阵。当前正式任务使用新的 `configs/structure_comparison.json`。

## 7. 验证状态

- 在合成数据上执行 14 项一轮训练，覆盖四个论文目标、三个教师的新方法和零目标消融。结果用于检查训练链路。见 [合成验证汇总](results/structure-verified/summary.csv)。
- 在 Wiki v2 前 4,000 条事件上，使用同一现有 TGN checkpoint 完成真实交互和零目标各一轮训练。采样候选及重新切分使该测试属于接口调试。两项均观测到 0.36% 的合法查询，说明路径覆盖率是正式实验需要追踪的量。见 [真实交互](results/structure-wiki-prefix-verify/full/result.json) · [零目标](results/structure-wiki-prefix-verify/zero/result.json)。
- 在同一 Wiki 前缀和 TGN checkpoint 上，四个论文组件也分别完成一轮 GCN 训练，辅助损失均有限。见 [四个论文组件验证日志](structure-wiki-paper.log)。
- 对完整 Wiki 训练流做了只读邻居缓存检查。110,232 条查询中有 10,870 条具备合法路径（约 9.86%），应用每批最多 4 条的预算后为 3,269 条（约 2.97%）。合法查询中有 8,607 条的至少一组包含重复历史事件。该检查复现事件读出槽，无模型训练或排名评估。见 [全量训练流覆盖率](results/structure-wiki-train-coverage.json)。
- 147 项回归测试通过，包括四视图交互符号、梯度、三教师因果读出、原有矩阵及新矩阵。
- 159 项正式运行计划已生成，四数据集全量性能表尚未运行。空白 MRR 应记录为待运行，不能用前缀分数或合成数据代填。

| 数据集 | 教师 | 学生 | 方法/结构目标 | 种子数 | 官方 Test MRR 均值 ± 标准差 | 合法查询率 | 耗时 | 峰值显存 |
|---|---|---|---|---:|---|---|---|---|
| Wiki v2 | TGN | GCN | `structure_interaction / temporal` | 待运行 | — | — | — | — |
| Wiki v2 | TGN | GCN | `structure_interaction / zero` | 待运行 | — | — | — | — |
