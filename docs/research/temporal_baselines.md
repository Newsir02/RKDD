# 时序图蒸馏论文与 TGN→GCN 对照

四项论文对照以固定 TGN 检查点、相同候选、单视图批前缀和同一个快照 GCN 为实验条件。代码复现各论文能迁移到当前任务的损失组件，统一替换骨干、原任务和输入协议。正式表格应写明「组件适配」及具体保留项。原论文的性能数字不可与适配结果直接比较。

| 运行名 | 论文与出版情况 | 原任务和输入 | 本项目使用的组件 |
|---|---|---|---|
| `dyngkd_structure` | Antaris et al., *Knowledge distillation on neural networks for evolving graphs*, Social Network Analysis and Mining, 2021 | 离散图快照、结构和时序注意力、有权链接预测 | F2 式（9）的结构特征 KL，将 TGN 读出节点特征对齐到 GCN 第一层 |
| `distill2vec_latent` | Antaris and Rafailidis, *Distill2Vec: Dynamic Graph Representation Learning with Knowledge Distillation*, ASONAM 2020 | 先离线预训练再在线蒸馏，使用快照序列的结构和时序注意力，预测后续链接 | 式（10）的最终节点嵌入 KL，保留节点特征分布监督 |
| `lightst_adaptive` | Zhang et al., *Efficient Traffic Prediction Through Spatio-Temporal Distillation*, AAAI 2025 | GNN→MLP 的多步交通流预测 | 作者训练器实际启用的预测 KL + 空间加权对比组件，按单个查询的同期候选集合计算 |
| `lkd4dytag_edge` | Roy et al., *LLM-driven Knowledge Distillation for Dynamic Text-Attributed Graphs*, AAAI 2025 AI4TS workshop | 冻结 LLM 提供边文本知识，轻量时序边 GNN 预测链接和边类别 | 式（7）、（16）–（18）的逐候选边 Hadamard 特征与负指数余弦距离 |

## 1. DynGKD 结构特征组件

DynGKD F2 对齐结构层 $\mathbf H$ 与时序层 $\mathbf B$。固定候选评分体系中的 GCN 只有空间消息传播层，使用式（9）的结构分支。学生特征 $s$ 取第一层 GCN 激活，教师特征 $t$ 取 TGN 时序邻居图注意力读出的节点嵌入。后者也含记忆信息，因此表中称为结构读出对齐。对源和所有候选节点出现位置分别计算

$$
\ell_{\mathrm{str}}=
 D_{\mathrm{KL}}\!\left(
 \operatorname{softmax}(P s/\tau)
 \;\Vert\;
 \operatorname{softmax}(t/\tau)\right).
$$

$P$ 为训练时线性对齐宽度，$\tau$ 为通用温度。以原文写出的学生与教师次序解释 KL，softmax 是将未归一化特征变为有效分布的实现约定。原论文还蒸馏中间时序嵌入，并有以权重重建误差为主的混合变体。本对照只使用结构支路与统一的链接 CE。论文作者所链接的官方仓库公开树目前仅列许可和说明文件，因此参照原文公式复现该支路。 [论文原文](https://link.springer.com/article/10.1007/s13278-021-00816-1#Sec10) · [论文所引仓库](https://github.com/stefanosantaris/DynGKD)

## 2. Distill2Vec 最终节点嵌入组件

原式（10）将学生任务损失与节点级 $KL(H^S\mid H^T)$ 结合。此处在每个查询的源节点及候选节点上，用上述相同的通道 softmax 约定和可学习 $P$ 计算 KL。此项使用 **GCN 最后一层**输出，DynGKD 结构组件使用第一层输出。至少两层的 GCN 保证这两个对照使用不同层，主实验固定该条件。适配沿用统一链接 CE 与历史前缀，原方法的双注意力学生、在线窗口、每个快照的训练和快照外推评估留在原任务内。 [论文式（10）](https://arxiv.org/pdf/2011.05664#page=4) · [作者代码](https://github.com/stefanosantaris/Distill2Vec)

## 3. LightST 加权空间对比组件

论文式（7）描述余弦对比，而作者仓库的现行 `train_epochs` 实际使用 `loss1 + 10*tkloss + 1*scl`。此处跟随公开算法的有效分支，保留预测分布 KL 和 `loss_cls`。把单个查询解释为一个同时刻，集合 $\mathcal V_q$ 包含该查询源及候选。对单位向量化后的同位置师生特征 $s_i,t_i$，令

$$
\begin{aligned}
p_i&=\exp\left((s_i^\top t_i)^2/0.05\right),\\
n_i^S&=\sum_{j\in\mathcal V_q}\exp\left(s_i^\top s_j/0.05\right),\qquad
n_i^T=\sum_{j\in\mathcal V_q}\exp\left(t_i^\top t_j/0.05\right),\\
\ell_{\mathrm{scl}}&=\operatorname{mean}_i
\tfrac12\left[\log(1+n_i^S/p_i)+\log(1+n_i^T/p_i)\right].
\end{aligned}
$$

如作者 `loss_cls`，两侧同域分母均包含对角项，正项乘对应余弦形成余弦平方。实现用 `logsumexp` 和 `softplus` 保持数值稳定。配置权重沿用主实验统一的 `kd_weight` 与 `relation_weight`，包含标准候选温度 KL，训练阶段 TGN 权重固定，不构造交通流区域或多时间维 TCN。 [AAAI 论文](https://ojs.aaai.org/index.php/AAAI/article/view/32096) · [作者训练器](https://github.com/lizzyhku/TP/blob/main/model/BasicTrainer.py#L100-L171)

## 4. LKD4DyTAG 边特征组件

如式（7）以源节点和候选节点的 Hadamard 乘积得到边表示 $e_M=h_M(u)\odot h_M(v)$。学生边表示经投影后，按式（16）–（18）的负指数余弦对齐：

$$
\ell_{\mathrm{edge}}=\operatorname{mean}_{(u,v)}
\exp\!\left[-\cos(Pe_S,e_T)\right].
$$

原方法以 LLM 的带上下文文本嵌入作为 $e_T$。统一对照用 TGN 的候选边表示替代文字教师，保留边级别的目标函数。当前 TGB 数据与 TGN 教师不接入论文中的文本、LLM 提示、训练标签提示及边分类任务。在原文中核实到 DTGB 数据仓库链接，但未核实到作者的训练代码仓库。 [论文与 workshop 状态](https://arxiv.org/abs/2502.10914) · [原文公式](https://arxiv.org/html/2502.10914v1) · [原任务数据 DTGB](https://github.com/zjs123/DTGB)

## 实现与评估口径

- [适配代码](../../loss/temporal_paper_adaptations.py) 与 [单元测试](../../tests/test_temporal_paper_adaptations.py) 覆盖四个目标、异宽师生训练投影梯度、教师无梯度以及 KL、LightST 对比和指数余弦的数值检验。
- 上述每项均使用同一未遮蔽历史前缀。受监督 GCN 的 CE 在各方法中保留。投影器只在训练计算损失使用，正常学生评估和推理不需要投影器。统一的权重起点可产生不同尺度的辅助损失，特别是 LightST 的对比温度为 0.05。正式性能比较应按共同验证预算为各基线选择蒸馏权重。
- 论文适配仅加入 TGN→GCN 主比较。不同教师对照固定为本课题的四视图结构交互蒸馏。 [实验方案](../../comparison_experiments.md)
- 已通过 7 项针对性测试。另在 48 个事件的合成数据上，使用项目 TGN、两层快照 GCN 和实际训练函数逐个运行了四项对照。四项损失均为有限值，辅助损失非零。此检查只验证执行路径，正式四数据集表现有待实际训练与评估。
