# 蒸馏方法调研与复现说明

核查日期：2026-11-14。任务：连续时间交互图上的候选链接预测，冻结时序教师向普通 GNN 学生迁移知识。

## 1. 调研范围、证据与筛选结论

检索关键词包括 Temporal GNN Distillation、Temporal Graph Knowledge Distillation、dynamic graph knowledge distillation、relational knowledge distillation，以及年份、会议和模型名组合。核查顺序为会议/期刊页面、论文原文、作者仓库；搜索引擎摘要仅用于发现线索。这里是可核验的定向系统调研，不声称穷尽所有论文。未找到官方仓库表示本次未核实，不能推断作者从未开源。

必须区分四类问题：

1. **连续时间交互图**：事件 `(u,v,t)`，历史前缀下预测后续交互，本项目属于此类。
2. **时序知识图谱（TKG）**：实体、关系、实体、时间四元组；跨语言对齐、关系类型和时间区间不能省略后仍称完整复现。
3. **时空预测**：固定传感器图上预测连续数值序列，目标通常是 MAE/MSE，不是候选链接分类。
4. **静态图或通用 KD**：可以作为迁移对照，但不是时序图蒸馏论文。

本轮未核实到同时满足“2023—2026、连续事件链接预测、时序 GNN 教师→普通 GCN/GraphSAGE、可直接移植完整算法”的代表性公开实现。下面不以改名的 KL 损失冒充该方向的新论文。实际新增的是可逐项验证的论文目标及明确标注的组件适配；完整时序方法因任务不一致或实现细节不足而不强行拼接。

### 1.1 时间相关图蒸馏论文

| 论文 | 年份、实际状态 | 教师→学生；知识与目标 | 原论文数据 | 代码与本项目决策 |
|---|---|---|---|---|
| [Mutually-paced Knowledge Distillation for Cross-lingual Temporal Knowledge Graph Reasoning](https://arxiv.org/abs/2303.14898)（MP-KD） | WWW 2023，arXiv 页面明确已录用；[出版论文](https://doi.org/10.1145/3543507.3583407) | 源语言 TKG 网络→目标语言 TKG 网络；伪实体对齐、时间跨语言注意力、随训练共同调整的知识迁移 | EventKG，12 个跨语言迁移任务 | 未核实官方仓库；缺少当前数据的跨语言实体对齐，不纳入主矩阵。不是普通 GNN 压缩 |
| [Spatio-Temporal Graph Knowledge Distillation](https://openreview.net/pdf?id=akKNGGWegr)（STGKD） | ICLR 2024 投稿稿；可检索 PDF 标为 under review。本轮 OpenReview 状态页遇到验证页，**不列为已录用 ICLR 论文** | 时空 GNN→MLP；时空信息瓶颈、teacher-bounded regression、空间/时间提示 | 稿件摘要称 3 个城市时空数据集；具体清单本轮未核实 | 未核实对应官方实现及正式发表版本；不实现不确定细节 |
| [Rethinking multi-level information fusion in temporal graphs: Pre-training then distilling for better embedding](https://www.sciencedirect.com/science/article/abs/pii/S1566253525002003)（ReMIT） | Information Fusion 121, 103127，2025 | node2vec 等预训练先验→时序图训练模块；邻域、社区、全图多层信息蒸馏，方向与本项目教师压缩不同 | 出版商可访问内容未提供可核验完整数据清单 | 未核实 2025 版本官方仓库；[DicTED 仓库](https://github.com/Samer-hue/DicTED)不能据名称直接当作该版本代码 |
| [Efficient Traffic Prediction Through Spatio-Temporal Distillation](https://arxiv.org/abs/2501.10459)（LightST） | AAAI 2025；[会议 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/32096/34251) | 高容量时空 GNN（MTGNN 系实现）→轻量 MLP；自适应蒸馏与时空对比目标，预测未来交通流 | PeMSD3/4/7/8、PeMSBAY，12 步输入/12 步输出 | [作者代码](https://github.com/lizzyhku/TP)；连续回归任务，不用候选 KL 替换其完整算法。项目旧 LightGNN 也不等于该论文 LightST |
| [FedDGL: Federated Dynamic Graph Learning for Temporal Evolution and Data Heterogeneity](https://proceedings.mlr.press/v260/xie25b.html) | ACML 2024 会议；PMLR 260 的出版年份为 **2025**，两者分别记录 | 联邦全局/客户端动态图模型；历史知识保留、全局原型正则、个性化聚合，不是固定单教师压缩 | [作者仓库](https://github.com/zach82/FedDGL)列出 Brain、DBLP3、DBLP5、Reddit 节点任务；其他任务未在此声称已复核 | 原文明确链接该仓库。缺少联邦客户端划分和聚合协议，不纳入单教师主矩阵；具体骨干超参需依据其 settings 复现 |
| [Compressing Transfer: Mutual Learning-Empowered Knowledge Distillation for Temporal Knowledge Graph Reasoning](https://ieeexplore.ieee.org/abstract/document/10852525)（MLEMKD） | IEEE TNNLS 2025；DOI 10.1109/TNNLS.2025.3525699 | 大 TKGR→小 TKGR；根据软标签置信度和熵变化筛选知识，再以师生认知差异调整教学 | 摘要报告 4 个基准；全文的数据名及骨干配置本轮未获取 | 未核实官方代码；不能仅凭摘要实现“等价版本”，因此未复现 |
| [LLM-driven Knowledge Distillation for Dynamic Text-Attributed Graphs](https://arxiv.org/abs/2502.10914)（LKD4DyTAG） | AAAI 2025 **AI4TS workshop**，不是 AAAI 主会论文 | LLM 驱动的邻域文本边表示→带时间边特征的 GATConv/TransformerConv；语义表示蒸馏与链接/边分类任务 | [原文](https://arxiv.org/html/2502.10914v1)使用 DTGB 的 6 个文本动态图，包含 Enron、ICEWS1819、Googlemap CT、GDELT、Stack Ubuntu 等；完整清单以原文附录为准 | 原文的 [DTGB 链接](https://github.com/zjs123/DTGB)是数据/基准代码，不能冒称方法官方仓库；当前数值特征数据无对应文本与 LLM 监督，不纳入主矩阵 |
| [Knowledge Distillation for Temporal Knowledge Graph Reasoning with Large Language Models](https://arxiv.org/abs/2601.00202) | 2026-01-01 arXiv v1；本轮未核实正式会议/期刊接收 | LLM 辅助高容量 TKG→低维 TKG；软分数及语义知识迁移 | 具体公式/配置未对该版本逐式复核，不与后续条目默认视为相同 | 未核实官方仓库；作为公开预印本记录，不进入实现清单 |
| [LLM-Guided Knowledge Distillation for Temporal Knowledge Graph Reasoning](https://arxiv.org/abs/2602.14428) | 2026-02-16 arXiv v1；页面无已核实的正式发表信息 | TTransE/TADistMult 高维教师及 LLM→低维 TKG 学生；分布、分数与语义相关目标；[原文](https://arxiv.org/html/2602.14428v1)给出教师 400 维、学生 25 维 | YAGO11k、WIKIdata12k | 未核实官方仓库；LLM 选择、知识生成及若干目标记号不足以重建可检验训练流程。与上一条作者高度重叠，但不擅自判定版本关系或计作两套独立方法 |
| [Interpretable Spatial-Temporal Forecasting via Additive Neural Decomposition and Knowledge Distillation](https://proceedings.mlr.press/v337/lee26d.html)（STGNAM） | UAI 2026，PMLR 337:3376–3397，正式论文集可核验 | D2STGNN 等黑箱时空教师→可加分解模型；预测软目标及可解释时空成分 | 论文页明确 METR-LA、PEMS-BAY；其余 3 项未从摘要补写 | 本轮未核实官方仓库；属于时空回归，未移植为交互图分类算法 |

另有 CIKM 2024 题为 *Distilling Knowledge Based on Curriculum Learning for Temporal Knowledge Graph Embeddings* 的线索（DOI `10.1145/3627673.3679896`）；本轮出版商全文未成功访问，骨干、数据和官方实现未完成核对，未将其作为已复现方法。动态图 **DistTGL** 是分布式训练系统，不能因名称以 Dist 开头就归为 KD；“graph distillation”也可能指图压缩而非教师学生蒸馏。

### 1.2 被选择的静态图/通用论文对照

| 方法 | 论文及年份 | 原始架构、数据 | 迁移理由与范围 |
|---|---|---|---|
| NOSMOG-RSD 组件 | [Learning MLPs on Graphs: A Unified View of Effectiveness, Robustness, and Efficiency](https://openreview.net/forum?id=Cs3r5KLdoj)，ICLR 2023 | GCN/GraphSAGE 等静态 GNN→MLP；Cora、Citeseer、Pubmed、Amazon Computers/Photo、ogbn-arxiv/products；[官方仓库](https://github.com/meettyj/NOSMOG) | 关系目标支持不同隐藏维度。仅迁移投影后 Gram 损失；不包含完整 NOSMOG 的 DeepWalk 位置编码、MLP 架构和对抗训练 |
| Logit Standardization | [Logit Standardization in Knowledge Distillation](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html)，CVPR 2024 | 图像 CNN/Transformer 教师学生，CIFAR-100/ImageNet；[官方仓库](https://github.com/sunshangquan/logit-standardization-KD) | 标准化预处理和 KL 可直接应用于同一候选集合的 logits；这是**通用 KD 对照** |
| RKD-D / RKD-A | Relational Knowledge Distillation，CVPR **2019**；[官方仓库](https://github.com/lenscloth/RKD) | 原始视觉度量学习架构/数据，不是时序图论文 | 年份在调研窗口外，作为经典关系对照；新的论文版使用 Huber，原项目 MSE 版仍保留 |
| DKD | Decoupled Knowledge Distillation，CVPR **2022**；[官方实现](https://github.com/megvii-research/mdistiller) | 图像分类教师学生，CIFAR-100/ImageNet | 年份在窗口外，作为经典输出对照；保留目标/非目标解耦和线性 warmup |

## 2. 核心公式和真实实现

每个查询有同一组 `C` 个候选，`y` 为正例索引，师生 logits 为 `z_t,z_s`；教师张量始终 detach。监督项为 `CE(z_s,y)`。学生和教师在同一历史前缀、同一候选上比较。新增代码位于 `loss/paper_baselines.py`；公共构造入口为 `make_objective`。原 `loss/distillation.py`、`loss/kd.py` 和 `loss/temporal_delta.py` 保持原有行为。

### Logits KD

\[
 p_t=\operatorname{softmax}(z_t/T),\qquad
 L_{KD}=T^2\frac1B\sum_b KL(p_t^{(b)}\Vert p_s^{(b)}),\qquad
 L=L_{CE}+\lambda_{KD}L_{KD}.
\]

`--method logits --temperature 2 --kd-weight 1`。复用既有 `logit_kd_loss`。

### Logit Standardization（新增，核心实现对齐）

\[
 Z(z)=\frac{z-\bar z}{s(z)+10^{-7}},\qquad
 s(z)^2=\frac{\sum_{c=1}^C(z_c-\bar z)^2}{C-1},\qquad
 L_{LS}=T^2 KL(\operatorname{softmax}(Z(z_t)/T)\Vert
                  \operatorname{softmax}(Z(z_s)/T)).
\]

`StandardizedLogitKD` 使用官方代码的**样本标准差**，不是默认改成总体标准差。参考 [固定提交 eb569ac 的 KD.py](https://github.com/sunshangquan/logit-standardization-KD/blob/eb569ac3783f2eed16fbc2b021101175af4fb2f2/mdistiller/distillers/KD.py)。迁移的是候选 logit 关系；图像增强、图像分类网络和原训练预算不迁移。

`--method logit_standard --temperature 2 --kd-weight 1`。测试与参考公式数值对齐，并检查常量 logits、反向传播、教师无梯度。未运行作者完整 CIFAR/ImageNet 实验。

### NOSMOG-RSD（新增，明确为组件适配）

取当前批次的源节点及候选节点表示，按相同顺序展开为 `H_s∈R^{N×d_s}`、`H_t∈R^{N×d_t}`，`N=B(C+1)`：

\[
 \widetilde H_s=\operatorname{ReLU}(H_sW+b),\qquad
 L_{RSD}=\frac1{N^2}\|\widetilde H_s\widetilde H_s^\top-H_tH_t^\top\|_F^2,\qquad
 L=L_{CE}+\lambda_{KD}L_{KD}+\lambda_RL_{RSD}.
\]

参考 [固定提交 7f8ef63 的 train_and_eval.py](https://github.com/meettyj/NOSMOG/blob/7f8ef63e955341f19e81107d1fc6e24f7e5763e3/train_and_eval.py) 与同提交 `models.py`。官方实现是**未归一化内积矩阵的 MSE**；不擅自换成 cosine 或归一化 Gram。投影层参加优化，保存于 `objective.pt`，推理学生不包含该层。

`--method nosmog_rsd --kd-weight 1 --relation-weight 1`。差异：原静态节点 minibatch 换成当前时刻查询节点的出现序列，重复节点保留重复权重；teacher/student 架构、损失组合权重采用本项目统一配置；投影 dropout 为 0；没有 DeepWalk 或 PGD。**结果必须标作 NOSMOG-RSD component adaptation，不能写 NOSMOG 官方完整复现。** Gram 计算为 `O(N²)`，正式运行需观察显存。

### RKD-D / RKD-A（新增论文版目标，保留旧控制项）

每个查询取源节点和候选，距离按正距离的均值归一化：

\[
 L_D=\operatorname{mean}_{i,j}\rho\!\left(
 \frac{\|s_i-s_j\|_2}{\mu_s}-\frac{\|t_i-t_j\|_2}{\mu_t}\right),
\]
\[
 L_A=\operatorname{mean}_{i,j,k}\rho\!\left(
 \left\langle\frac{s_j-s_i}{\|s_j-s_i\|},\frac{s_k-s_i}{\|s_k-s_i\|}\right\rangle-
 \left\langle\frac{t_j-t_i}{\|t_j-t_i\|},\frac{t_k-t_i}{\|t_k-t_i\|}\right\rangle\right),
\]

其中 `ρ=smooth_l1`、beta=1。`paper_rkd` 与 [官方 metric/loss.py](https://github.com/lenscloth/RKD/blob/master/metric/loss.py)一致使用 Huber，并在最终均值中保留对角/退化位置；距离加数值下界。关系在每个查询内部计算后对批次平均，属于采样作用域的适配。旧 `loss/kd.py` 的 MSE 关系控制项仍供旧入口和 TRD 使用。

`--method rkd_distance` 或 `--method rkd_angle`，总目标 `CE + λKD KL + λR RKD`。这不是论文原始视觉度量学习训练策略的完整复现。

### DKD（独立论文版公式与可配置原策略）

令 `b(p)=[p_y,1-p_y]`，`q` 为排除正例后的概率分布：

\[
 L_{DKD}=T^2[\alpha KL(b(p_t)\Vert b(p_s))+\beta KL(q_t\Vert q_s)],
\]
\[
 L=L_{CE}+\lambda_{KD}\min(e/E_{warm},1)L_{DKD}.
\]

`--method dkd --dkd-alpha 1 --dkd-beta 8 --dkd-warmup 20`；0 表示关闭 warmup。`paper_dkd` 与 [官方 DKD.py](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py)核对，**移除目标类别后重新归一化非目标分布**。旧 `loss/kd.py` 的实现只将目标概率清零，未做该归一化，因此新比较入口不调用它，旧入口行为仍保留。数值实现用精确移除代替 -1000 屏蔽，用 logsumexp 稳定计算二分类概率；有限正常输入下等价于原公式。已增加两候选/多候选参照测试，避免把非归一化 KL 当成 DKD。温度默认 2，为当前统一候选任务配置，不宣称照搬原图像实验超参。正类是本次交互的目标候选，不是固定语义类。

### 用户 TRD（原实现直接调用）

\[
 r(u,v;H)=\cos(h_u(H),h_v(H)),\quad
 \Delta r=r(u,v;H_{k-1})-r(u,v;H_{k-2}),
\]
\[
 L_{TRD}=CE+\lambda_{KD}L_{KD}+\lambda_R\|\Delta r_s-\Delta r_t\|^2.
\]

`--method trd --trd-mode temporal`。调用原 `TemporalDeltaDistillation` 与 `train_temporal_delta_epoch`；首批没有两个前缀时关系项为零。`static/shuffled/reverse/zero` 消融直接沿用原接口。本轮未修改 TRD 公式、双前缀更新次序或原 `PlainGNNStudent`。

## 3. 接口和命令

新比较入口为 `benchmark.py`，不会改变 `train.py`、`train_trd.py` 的原命令。每个运行目录只能写一次；需要重跑时换目录。

```bash
# 同一套配置先训练教师
python benchmark.py --stage teacher --dataset synthetic --quick --teacher-arch tgn --output-dir results/demo-teacher
# 然后选择任一方法；教师配置、种子、候选数和 batch size 必须匹配
python benchmark.py --stage distill --dataset synthetic --quick --teacher-arch tgn --student-model sage --method logit_standard --teacher-checkpoint results/demo-teacher/teacher.pt --output-dir results/demo-lskd
python benchmark.py --stage distill --dataset synthetic --quick --teacher-arch tgn --student-model gcn --method nosmog_rsd --teacher-checkpoint results/demo-teacher/teacher.pt --output-dir results/demo-rsd
```

JSON 单次配置使用 `--config path.json`，命令行覆盖配置；未知字段/非法方法拒绝执行。矩阵配置和完整流程见 `comparison_experiments.md`。

## 4. 验证、修复和边界

- 已执行官方公式参照测试、常量输入测试、不同隐藏维度反向传播、教师梯度隔离、状态因果性、checkpoint roundtrip、协议不匹配拒绝检查。
- 已完成合成数据 141 次训练，47 个 3-seed 分组。输出在 `results/verified-smoke/`；仅验证执行链，不能用这些 tiny 数据上的分数论证方法优越性。
- 完整回归在项目隔离的 `py-tgb==2.2.0` 依赖下执行。验证计数与日志见 `docs/research/validation.md`。
- 新学生加入普通可学习节点 ID 表示，避免全零属性下 GraphSAGE 把所有节点映射为常量；这是传导式设定，不能声称严格归纳性能。
- 验证/测试均在固定权重下重放历史，不沿用训练过程中不同权重形成的状态。教师文件在蒸馏前后计算 SHA256，若发生变化则报错。
- 原官方评估测试仍断言逐条打分，已改为分别检查分块 1/16，并保持全部打分后才更新真实边的约束；评估实现未改动。
- 无真实数据完整实验、无原论文排行榜数值复现、无 GPU 显存实测。本文中未列出的论文细节保持“未核实”，不以推测填补。
