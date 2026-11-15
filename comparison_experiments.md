# 统一对比实验实施方案

日期：2026-11-14。当前交付是可运行代码、完整计划矩阵和合成数据验证；**5 个真实数据集上的完整性能实验尚未执行**。论文方法来源与适配边界见 `distillation_methods.md`。

## 1. 五个数据集

统一选择 TGB 动态链接预测数据，主指标均为官方 MRR，避免混用二分类 AP 与不同负例数的排名结果。统计引自 [TGB 官方动态链接预测页面](https://tgb.complexdatalab.com/docs/linkprop/)，仅作为下载和容量规划参考。固定 `py-tgb==2.2.0` 的本地元数据将 Wiki、Review、Coin、Flight 映射到 v2，Comment 映射到 v1。官网部分表格省略版本后缀，因此**最终实验以下载包版本和本地实际统计为准**，不能仅依据展示名称认为版本相同。

| CLI 数据名 | 固定包的版本 | 官方页面节点数 | 官方页面交互数 | 特点 | 主指标 |
|---|---|---:|---:|---|---|
| `tgbl-wiki` | v2 | 9,227 | 157,474 | 用户—页面编辑，二部图，边含文本衍生特征 | MRR |
| `tgbl-review` | v2 | 352,637 | 4,873,540 | 长跨度商品评价，二部图，稀疏新交互 | MRR |
| `tgbl-coin` | v2 | 638,486 | 22,809,486 | 加密资产转账，突发交易与反复交互 | MRR |
| `tgbl-comment` | v1 | 994,790 | 44,314,507 | Reddit 用户回复，有向社交交互 | MRR |
| `tgbl-flight` | v2 | 18,143 | 67,169,570 | 航班活动，大量交互、较少不同时间步 | MRR |

数据通过既有 `dataset/tgb_temporal.py` 与 PyG TGB 包装器加载；不重排事件、不自行重新划分官方 train/val/test。使用原库的验证/测试负例文件。预处理脚本会加载两份负例以验证可用性，记录版本、实际节点/事件/切分数、数据指纹以及找到的负例文件 SHA256。原 UCI、Enron、synthetic 等加载器保留，可作附加实验，但不冒称其协议与主表完全相同。

```bash
python -m experiments.prepare_data --datasets tgbl-wiki --data-root datasets --download --output results/wiki-data.json
# 首个数据集确认后再准备其余四个；总数据量大，需预留磁盘和内存
python -m experiments.prepare_data --datasets tgbl-review tgbl-coin tgbl-comment tgbl-flight --data-root datasets --download --output results/other-data.json
```

已有数据时去掉 `--download`，此时禁止下载。包缺失或数据不完整会报错，不会自动用 synthetic 冒充真实实验。

## 2. 三个教师与两个学生

### 教师

复用现有实现，**不是重新下载的论文官方骨干**。建议论文中标明 project implementations，并在未来主张复现原论文教师性能前增加独立官方骨干校验。

| `--teacher-arch` | 代码 | 机制及当前限制 | 统一默认值 |
|---|---|---|---|
| `tgn` | `model/teacher.py`、`model/tgn.py` | PyG TGN memory、消息更新、时间编码及查询聚合 | hidden=64，time=16，邻居=24，teacher-layers=1 |
| `tgat` | `model/tgat_teacher.py` | 现有时间注意力查询实现；不宣称等价于原论文所有递归多层 TGAT 设定 | hidden=64，time=16，邻居=24，heads=2 |
| `graphmixer` | `model/graphmixer_teacher.py` | 现有邻居混合与时间编码实现；不宣称官方 GraphMixer 结果复现 | hidden=64，time=16，邻居=24 |

用已有 GraphMixer 替代候选 DyGFormer，是为保留经过项目回归的三个教师，避免本轮引入未经验证的第四套复杂骨干。它提供与递归记忆、注意力不同的时序表示。三个教师具有不同固有结构；隐藏维度一致不代表参数量相同，结果中单独统计参数。

### 普通 GNN 学生

新增 `model/snapshot_gnn.py::SnapshotGNN`。两者均无时间编码、RNN、TGN 记忆和时间注意力；共享边打分 MLP。普通可学习节点 ID 表示加静态属性线性投影作为输入，因而明确为**传导式**设定。

| `--student-model` | 图卷积 | 默认超参 |
|---|---|---|
| `gcn` | PyG `GCNConv`；归一化权重由当前快照度数提供，含自环 | hidden=32，2 层，邻居预算 K=8，ReLU，dropout=0 |
| `sage` | PyG `SAGEConv`，mean 聚合及根节点线性变换 | 同上 |

数据流处理器维护已观察的每节点最近 K 次交互；神经网络只看到该快照的邻接，不看事件时间或消息特征。这是**在有界历史快照上的普通 GNN**，不是所有历史边的 full-batch GCN。重复交互保留为平行消息；双向插入后各节点独立保留 K 条，因此截断快照可能非对称。GCN 使用该快照上的对称形式度归一化。按查询节点展开 L 跳计算图，归一化度来自快照，不随候选分块改变。边的历史选择策略也是实验协议的一部分，K 必须跨方法相同。

没有把原 `PlainGNNStudent` 更名为 GCN/GraphSAGE；原学生与入口继续保留。ID embedding 的参数量计入总参数，百万节点时其内存可能主导学生成本，不应只报告卷积层参数。

## 3. 方法与训练配置

| 实验项 | CLI | 训练目标 | 论文命名要求 |
|---|---|---|---|
| 教师独立训练 | `--stage teacher` | CE | project TGN/TGAT/GraphMixer |
| 学生独立训练 | `--stage student --method supervised` | CE | GCN 或 GraphSAGE supervised |
| Logits KD | `--stage distill --method logits` | CE + KL | 通用 KD |
| DKD | `--method dkd` | CE + 解耦 KL，warmup | 原公式迁移到候选任务 |
| RKD-D/A | `--method rkd_distance` / `rkd_angle` | CE + KL + Huber 关系项 | 查询内采样 RKD；与旧 MSE 版区分 |
| Logit Standardization | `--method logit_standard` | CE + 标准化 logits KL | 通用 KD，CVPR 2024 |
| NOSMOG-RSD | `--method nosmog_rsd` | CE + KL + 投影 Gram MSE | **仅 NOSMOG 关系组件适配** |
| TRD | `--method trd` | 原 CE + KL + 双前缀有符号关系增量 | 用户方法；核心代码未改 |

默认预算：AdamW；teacher/student LR=0.001；weight decay=0；batch size=128；训练候选=16（1 正例+15 负例）；最多100 epoch；patience=20；梯度裁剪=5；种子42/43/44；温度2；KD/关系权重1；dropout=0。DKD α=1、β=8，20 epoch线性 warmup。默认系数是**待验证的统一起点，不是调参后的最优配置**。

正式论文应给所有方法相同验证集调参预算，例如每方法最多12组，候选温度 `{1,2,4}`、KD权重 `{0.5,1,2}`，关系权重 `{0.1,1,10}`，在预先固定的搜索子集中比较。最终报告搜索成本；不按测试结果选择超参。若改 batch size、候选数、教师训练预算，须重训配套教师并使用新结果目录。

所有学生方法每 epoch 使用相同真实事件和相同候选随机序列。TRD需要额外的历史前缀前向，所以“同 epoch预算”不等于同墙钟预算；同时报告耗时，并补充固定训练秒数预算的敏感性实验。早停允许 epoch 数不同，日志保留实际运行数和最佳轮次。

## 4. 可见性、数据泄漏与公平控制

1. **可见性**：按 `(timestamp,event_index)` 的已给定顺序流式处理；批次内所有查询共享批前历史，然后才写入本批正边。时间相同的事件按文件 event_index 顺序处理，跨批相同 timestamp 的先前事件可见。若任务要求严格 `t'<t`，须另做按 timestamp 分组协议，不能把当前结果标成该协议。
2. **教师蒸馏**：训练集阶段仅查询训练前缀；冻结参数且禁用 dropout。没有缓存未来事件的表示，也不在学生训练中查询验证/测试教师标签。
3. **时间归一化**：新增入口只用训练集首尾确定尺度；验证/测试允许越过1，禁止用测试末时刻拟合尺度。
4. **验证与测试**：每次验证用当前固定权重清空并重放训练历史；最佳权重测试前重放训练+验证历史。测试期间在线写入已评分的真实交互属于明确的 online evaluation 协议。测试分数不参与早停。
5. **候选协议**：训练复用现有 destination domain，新 `dataset/benchmark_loader.py` 在该域中均匀无放回采样，排除源节点和当前正例，正例位置均匀随机。使用独立 Python RNG，每轮重置固定种子；采样分布与旧入口一致，具体随机序列不同，避免逐事件构造全域随机排列。所有新实验共用此采样器，协议名为 `batch-prefix-v2-uniform-sample-train-time-scale`。已知整个数据集节点/目的节点 ID 域属于传导式元数据，不等于看见未来边。验证测试使用 TGB 官方固定负例；smoke 使用固定采样候选，永远记录 `official=false`。
6. **教师相同**：每 dataset/teacher/seed 只训练一个 checkpoint，在两种学生和全部蒸馏方法间共享。训练前严格检查数据指纹、协议、教师架构、种子、batch/candidates和预算；前后校验教师文件 SHA256。
7. **学生相同**：加载教师之后重新设置学生初始化种子；同一学生架构和种子的不同方法拥有相同初始学生权重。NOSMOG训练投影单独记参数，推理剥离。
8. **结果隔离**：汇总按数据集、版本、实际数据指纹、骨干、方法、配置、源码、软硬件环境及官方/非官方协议分组。数据指纹不含模型种子，但包含实际张量和切分。合成数据按seed生成不同事件流，是唯一有意跨不同数据指纹汇总的smoke例外。重复seed报错，缺种子标 incomplete，单seed标准差为空，不填0。
9. **保留旧实验**：新入口不重写旧 TRD 和已有 Baseline；旧 checkpoint 为避免协议混淆不会被自动当作新 benchmark 教师使用。旧入口仍可正常加载它们。

CUDA 的 scatter/reduce 等算子在不同设备或并行调度下可能存在浮点差异；固定种子不等于跨 GPU 字节级一致。运行记录包含设备和版本，GPU与CPU结果分开汇总，正式结论依靠多种子统计。

## 5. 组合矩阵与启动

| 数据集 | TGN→GCN/SAGE | TGAT→GCN/SAGE | GraphMixer→GCN/SAGE |
|---|---|---|---|
| Wiki v2 | 7 方法×3 seeds×2学生 | 同左 | 同左 |
| Review v2 | 同上 | 同上 | 同上 |
| Coin v2 | 同上；真实规模未跑 | 同上 | 同上 |
| Comment v1 | 同上；真实规模未跑 | 同上 | 同上 |
| Flight v2 | 同上；真实规模未跑 | 同上 | 同上 |

形式上共30个 dataset/teacher/student组合。`configs/comparison.json`生成45次教师训练+30次监督学生训练+630次蒸馏=**705次运行**。监督学生与教师无关，因此不重复训练三遍。该矩阵说明接口可组合，不是声称大数据组合已通过资源容量验证。

```bash
# 默认只写 manifest，不训练
bash scripts/run_comparison.sh --config configs/comparison.json --output-root results/main
# 确认已准备数据和 GPU 后执行；顺序运行，先教师再学生
bash scripts/run_comparison.sh --config configs/comparison.json --output-root results/main --execute
# 跳过配置、源码和 checkpoint 字节均匹配的成功运行
bash scripts/run_comparison.sh --config configs/comparison.json --output-root results/main --execute --resume
bash scripts/summarize.sh results/main --output results/main/summary.csv
```

PowerShell 等无 Bash 环境直接用 `python -m experiments.matrix`、`python -m experiments.summarize`；`scripts/run_single.sh`透传参数到 `benchmark.py`。可以通过 `PYTHON=/path/to/python`指定解释器。`--max-runs N`限制本次执行数；失败写入`failure.json`并停止，半成品不会冒充成功。修正后选新目录；`--resume`不恢复优化器或中断epoch，只跳过完整成功运行。

## 6. 日志、模型与效率指标

每次运行目录包含：

- `job.json`（矩阵任务）、`config.json`（实际生效配置）、`epochs.jsonl`（每轮损失/验证/训练时间）。
- `teacher.pt`或`student.pt`：最佳验证轮权重、配置和数据指纹。`objective.pt`只保存蒸馏训练辅助投影。
- `result.json`：测试指标、最佳/实际epoch、数据统计和版本、模型及教师SHA256、源码指纹、Python/依赖/设备信息。
- 参数量区分模型 trainable/all/runtime-state 与 training auxiliary；训练时间包含教师前向和学生训练，不含验证；验证时间另列。
- `cuda_peak_allocated_bytes`与`cuda_peak_reserved_bytes`记录训练期间整个进程的 CUDA 峰值（含教师），每个矩阵任务开始前清理已释放的CUDA缓存，减少前一个任务对reserved值的影响。CPU运行写`null`，不是0MB。
- 推理延迟在最佳权重和固定 pre-test 历史快照上测量，先预热并同步CUDA，排除数据传输和状态更新。延迟候选数使用统一训练候选数；**它不是官方多负例评估总耗时**。同时记录实际 batch 数，避免 tiny 数据的测量被误解为正式吞吐。

大图上全部事件特征、邻居队列和节点 ID 表可能占用显著内存。运行前先做容量检查；若OOM，降低batch/hidden/K时必须对该数据集的所有方法一致，并重建配套教师，单独汇总配置。不能只为一个方法变更设置再并入原表。

## 7. 消融及额外实验

- TRD目标：`--trd-mode temporal/static/shuffled/reverse/zero`；`--kd-weight 0`（无logits）；`--relation-weight 0`（只logits）。原RKD MSE控制通过旧入口单独标明，不与新Huber RKD重名。
- 学生容量：hidden `{16,32,64}`，层数 `{1,2,3}`，历史K `{4,8,16}`；教师checkpoint共享。
- 计算公平性：报告固定epoch和固定墙钟预算两组，后者需要额外实现时间停止控制或在日志中选择预注册预算内的验证checkpoint，不能根据测试挑选。
- 可见性：batch-prefix主协议与严格timestamp分组协议作为独立研究；后者本轮未实现。
- 稳健性：三个种子为最低要求；关键结论建议五个种子并作配对比较。数据集间结果不直接平均不同指标/协议。
- 泛化：节点ID输入意味着当前主矩阵非严格inductive；若移除ID需为无属性数据提供合法静态属性，并独立报告，不得隐瞒设定变化。

## 8. 结果表模板

下表均为空值，待真实实验填充。完整30组合模板已生成到 `docs/research/results_template.csv`。

| Dataset/version | Teacher | Student | Method | n seeds | Test MRR mean±sd | Params | Train s | Query ms | Peak CUDA MiB | Protocol |
|---|---|---|---|---:|---|---:|---:|---:|---:|---|
| Wiki/v2 | TGN | GCN | logits | 待运行 | — | — | — | — | — | batch-prefix，官方负例 |
| Wiki/v2 | TGN | GCN | logit_standard | 待运行 | — | — | — | — | — | 同上 |
| Wiki/v2 | TGN | GCN | nosmog_rsd component | 待运行 | — | — | — | — | — | 同上 |
| Wiki/v2 | TGN | GCN | TRD | 待运行 | — | — | — | — | — | 同上 |

样本标准差使用`ddof=1`。AP/AUC只有在另设明确定义的二分类采样任务时才添加；本次5个TGB主任务遵循官方MRR，不为增加指标而把不同负采样下的AP混入主表。

## 9. 从零复现流程及实际完成状态

```bash
# 推荐独立 Python 3.10 环境；CUDA PyTorch wheel 需与服务器驱动匹配
python -m pip install -r requirements-benchmark.txt
python -m pytest -q
python -m experiments.matrix --config configs/smoke.json --output-root results/my-smoke --execute
python -m experiments.summarize results/my-smoke --output results/my-smoke/summary.csv
python -m experiments.prepare_data --datasets tgbl-wiki --data-root datasets --download
python -m experiments.matrix --config configs/comparison.json --output-root results/main
python -m experiments.matrix --config configs/comparison.json --output-root results/main --execute
python -m experiments.summarize results/main --output results/main/summary.csv
```

本机测试解释器为`D:\Anaconda\envs\DGL\python.exe`；基础环境有torch2.1.0/PyG2.6.1，额外将py-tgb2.2.0安装至项目`.benchmark-deps/`，通过`PYTHONPATH`启用，没有修改系统环境。实际日志和校验记录见`docs/research/validation.md`。

已完成：代码实现、静态编译、完整回归、141次CPU合成训练、47组3-seed汇总、705任务正式清单、断点跳过检查。未完成：五份真实数据下载及实际统计核对、真实规模GPU运行、主表填值、原论文完整排行榜复现、超参数搜索、严格timestamp组协议。当前结果不能证明TRD胜过论文方法，也不能用合成任务分数替代论文结果。
