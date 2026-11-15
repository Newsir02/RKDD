# 时序关系增量蒸馏（TRD）

## 研究问题

在连续时间链接预测中，教师对同一节点对的表示关系会随已观察事件更新。TRD 研究这些变化中有多少能被普通 GNN 从邻居快照中学到。主实验假设是：在相同学生、教师和候选采样下，有正确节点对对应的关系增量，比单纯匹配 logits 或静态关系提供更多预测信息。

经典 RKD 迁移表示之间的距离和角度。TRD 在相同查询上重复测量节点对关系，迁移新增历史引起的变化。LLP 依据锚节点匹配候选排序和分布，可以作为候选关系蒸馏的参照。动态教师学习能力和多视图关系蒸馏都已有研究，TRD 的研究贡献聚焦于真实有序历史下的增量监督及实际可迁移范围。 [RKD](https://arxiv.org/abs/1904.05068) · [LLP](https://arxiv.org/abs/2210.05801) · [Distill2Vec](https://arxiv.org/abs/2011.05664) · [VRM](https://arxiv.org/abs/2502.20760)

## 模型与信息边界

教师为冻结的 TGN、TGAT 或 GraphMixer。给定相同的查询 $q=(u,\mathcal C_q,t_q)$，分别使用历史 $H^-$ 和补入上一批已观察事件后的历史 $H^+$，得到教师评分主干中的源节点和候选节点表示。训练期间教师参数固定，事件状态按顺序更新。新训练的 TGAT 和 GraphMixer 从训练前缀确定时间归一化尺度。 [TGN](https://arxiv.org/abs/2006.10637) · [TGAT](https://arxiv.org/abs/2002.07962) · [GraphMixer](https://arxiv.org/abs/2302.11636)

学生用同一个普通 GNN 编码两份图快照。$\mathcal N_K(v;H)$ 是历史 $H$ 下节点 $v$ 的最近 $K$ 次交互对应的邻居多重集。对静态节点特征 $x_v$ 和可学习的节点 ID 表示 $e_v$，计算

$$
b_v=f_\theta(x_v)+e_v,\qquad
m_v(H)=\frac{\sum_{w\in\mathcal N_K(v;H)} b_w}{\max(1,|\mathcal N_K(v;H)|)},\qquad
h_S(v;H)=\operatorname{ResMLP}_\theta\bigl(b_v+g_\theta(m_v(H))\bigr).
$$

最近邻选择在输入图的构造阶段进行。普通学生的前向只读取邻居、节点 ID 和静态特征，聚合对邻居排列不敏感。评分头接收源和目标表示、逐元素乘积及绝对差。最近 $K$ 缓存可能淘汰旧边，因此完整事件历史满足前缀包含关系，缓存中的邻居边集合可以发生替换。

## 目标函数

对教师或学生 $M\in\{T,S\}$，以 $\epsilon=10^{-8}$ 对表示做稳定归一化 $\bar h=h/\max(\|h\|_2,\epsilon)$，定义

$$
R_M(q,v;H)=\langle \bar h_M(u;H),\bar h_M(v;H)\rangle,
\qquad
\Delta R_M(q,v)=R_M(q,v;H^+)-R_M(q,v;H^-).
$$

增量的正负表示节点对在表征空间中的相似度增强或减弱。对非零且范数至少为 $\epsilon$ 的表示，$\Delta R_M=-\tfrac12\Delta\|\bar h_u-\bar h_v\|^2$。因此该目标对应归一化表示的成对距离增量，师生可以使用不同的隐藏维度。若关系可分解为 $R(u,v;H)=A(u,v)+B(u,v;H)$，差分会消掉与历史无关的 $A(u,v)$。

设 $B$ 为查询数，$C$ 为每条查询的候选数，$z_{M,q}^+$ 为在 $H^+$ 下的候选 logits，$y_q$ 为真实候选索引。三个目标分别为

$$
\mathcal L_{\mathrm{task}}=-\frac{1}{B}\sum_q \log\operatorname{softmax}(z^+_{S,q})_{y_q},
$$

$$
\mathcal L_{\mathrm{logits}}=\frac{\tau^2}{B}\sum_q
D_{\mathrm{KL}}\!\left[\operatorname{softmax}(z^+_{T,q}/\tau)\,\Vert\,
\operatorname{softmax}(z^+_{S,q}/\tau)\right],
$$

$$
\mathcal L_{\mathrm{RKD}}=\frac{1}{BC}\sum_q\sum_{v\in\mathcal C_q}
\left[\Delta R_S(q,v)-\operatorname{sg}\bigl(\Delta R_T(q,v)\bigr)\right]^2.
$$

$$
\mathcal L = \mathcal L_{\mathrm{task}}+\lambda_1\mathcal L_{\mathrm{logits}}+\lambda_2\mathcal L_{\mathrm{RKD}}.
$$

教师项停止梯度。两次学生关系前向都参与反向传播。当前实现的初始设置为 $\tau=2$、$\lambda_1=1$、$\lambda_2=1$，TRD 在训练首批缺少双前缀时将关系项设为零。静态关系、RKD 距离和角度只在对应的对照设置中替换关系项。距离和角度对照在每条查询的源节点和候选集合上计算，是对原始 RKD 的适配。 [RKD](https://arxiv.org/abs/1904.05068)

## 训练和评测协议

对于查询批 $B_k$，首先用截至 $B_{k-2}$ 的历史前向，写入 $B_{k-1}$ 的正事件，再对同一批 $B_k$ 查询前向。之后计算三个目标并更新学生参数一次。两次前向复用查询时间、候选节点和学生参数。$B_k$ 的真实边在后续步骤才进入历史。数据有相同时间戳时使用已有事件顺序。此构造测量固定查询时刻对新增历史的响应。

推理使用当前历史快照和一次学生前向。历史缓存仍需更新。关系算子针对一批源节点与候选节点的时间和中间表示开销随候选数线性增长。TRD 训练多出一次教师和学生前向，训练显存与耗时需实测。验证基于训练历史。测试重放完整的训练与验证历史。每个批次先打分，再写入刚观察的边。TGB 数据沿用其评测负样本。 [TGB](https://arxiv.org/abs/2307.01026)

## 可学性和可能的失败

令 $X$ 为学生在两个历史前缀看到的图快照与节点输入，$Y=\Delta R_T$。平方误差下，任意仅依赖 $X$ 的预测器的最小期望误差包含 $\mathbb E[\operatorname{Var}(Y\mid X)]$。同一个学生在两份完全相同的快照上给出相同关系，故 $\Delta R_S=0$。教师因时间间隔、消息或更长记忆产生的变化可能超出学生可观测范围。

余弦变化描述教师潜在几何的响应，与预测 MRR 的联系需要验证。若教师目标的增量普遍接近零，收益也可能仅来自平滑约束。若加入关系项后 MRR 降低，先检查关系增量尺度、教师质量、学生快照变化比例和 $\lambda_2$，并按可观测性分层报告结果。

## 实验计划

1. 主对照采用相同普通学生和被冻结的同一教师，同时报告监督、Logits KD、静态余弦、RKD-D/A、TRD。比较 TRID、DKD 和排序 KD 时，注明 `LightGNNStudent` 与普通学生的架构差异。
2. 对 `temporal`、`shuffled`、`reverse`、`zero` 保持相同优化预算和负例。打乱目标对应可检验节点对身份的作用，反向检验符号，零增量检验平滑，静态对照检验历史差分。
3. teacher checkpoint、数据和候选采样固定，学生以配对种子运行。建议至少五个种子，在验证集选择 $\lambda_1$、$\lambda_2$ 和温度，再固定设置报告测试 MRR、成对差值及不确定性。
4. 分别记录训练每 epoch 耗时、峰值显存、参数与缓存大小、相同批次及候选数下的推理延迟。模型前向计时和包含缓存更新的端到端延迟分别报告。

合成数据冒烟测试用于验证训练、加载与评测可运行。真实数据上的效果由上述实验确定。
