# DeepSeek-V3 学习笔记

> **技术报告：** [《DeepSeek-V3 Technical Report》](https://arxiv.org/abs/2412.19437)
>
> DeepSeek-V3 总参数 671B、每 token 激活 37B（671B-A37B），支持 128K 上下文。
>
> **架构上 V3 沿用 V2 的 MLA 与 DeepSeekMoE**，V3 的核心创新点：
>
> - 无辅助损失负载均衡（Auxiliary-Loss-Free）
> - Multi-Token Prediction（MTP）
> - 训练 Infra（FP8 混合精度训练 + DualPipe）

## 1. 模型架构

| 模型结构 | 模块展开 |
| --- | --- |
| ![](https://cdn.nlark.com/yuque/0/2026/png/26585671/1793008881158-eb5a539c-c71c-46b1-94ff-4dcb8216542b.png) | ![](https://cdn.nlark.com/yuque/0/2026/png/26585671/1793008889722-2a24402c-2feb-4f6f-8f25-05616b531c09.png) |

> 其中的 MLA 和 FFN MoE 的架构仍然沿用 V2 版本：[DeepSeek-V2](https://swfvqxo30ma.feishu.cn/wiki/PFQDwdWb4ivhtUkKTfPcBjXvnMd)，但是对 FFN MoE 的平衡策略进行了更新。

![](https://cdn.nlark.com/yuque/0/2026/png/26585671/1793008927752-ed720870-c45f-4f8f-b71c-a0d59c8303f7.png)

### 1.1 DeepSeekMoE

DeepSeek V3 的 FFN 沿用 V2 的 DeepSeekMoE（细粒度专家 + 共享专家）。令 $\mathbf{u}_t$ 是第 $t$ 个 token 的 FFN 输入，输出 $\mathbf{h}_t'$ 为：

$$
\mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{N_s} \operatorname{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t}\, \operatorname{FFN}_i^{(r)}(\mathbf{u}_t)
$$

$$
\begin{aligned}
&g_{i,t} = \frac{g_{i,t}'}{\sum_{j=1}^{N_r} g_{j,t}'},
\\ & g_{i,t}' = \begin{cases} s_{i,t}, & s_{i,t} \in \operatorname{Topk}(\{s_{j,t} \mid 1 \leq j \leq N_r\}, K_r) \\ 0, & \text{otherwise} \end{cases},
\\ & s_{i,t} = \operatorname{Sigmoid}(\mathbf{u}_t^{\top} \mathbf{e}_i)
\end{aligned}
$$

其中 $N_s$、$N_r$ 是共享（shared）/ 路由（routed）专家数， $K_r$ 是每 token 激活的路由专家数， $\mathbf{e}_i$ 是第 $i$ 个路由专家的中心， $s_{i,t}$ 是 token 到专家的亲和度（token-to-expert affinity）。

> 回顾 [DeepSeek-V2](https://my.feishu.cn/wiki/PFQDwdWb4ivhtUkKTfPcBjXvnMd?from=from_copylink) 的 DeepSeekMoE 公式：
>
> $$
> \mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{N_s} \operatorname{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t}\, \operatorname{FFN}_i^{(r)}(\mathbf{u}_t)
> $$
>
> $$
> \begin{aligned}
> &g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \operatorname{Topk}(\{s_{j,t} \mid 1 \leq j \leq N_r\}, K_r) \\ 0, & \text{otherwise} \end{cases},
> \\ & s_{i,t} = \operatorname{Softmax}_i(\mathbf{u}_t^{\top} \mathbf{e}_i)
> \end{aligned}
> $$
>
> **对比之下，V3 的 DeepSeekMoE 与 [DeepSeek-V2](https://my.feishu.cn/wiki/PFQDwdWb4ivhtUkKTfPcBjXvnMd?from=from_copylink) 的区别**：
>
> - 亲和度（token-to-expert affinity）计算从 V2 的 Softmax 改成 Sigmoid。
> - 门控值 $g_{i,t}$ 在被选中的专家间再做一次归一化（除以 $\sum g_{j,t}'$）。
> - V3 配置为每个 MoE 层 1 个共享专家 + 256 个路由专家、专家中间维 2048、每 token 激活 8 个路由专家，前 3 层为 dense FFN、其余为 MoE。V2 配置每个 MoE 层是 2 个共享专家 + 160 个路由专家，专家中间维度 1536，每 token 激活 6 个路由专家，除第 1 层外所有 FFN 都替换为 MoE。

### 1.2 无辅助损失负载均衡（Loss-Free Balancing）

**动机：** MoE 负载不均会引发路由坍缩、并在专家并行下降低效率。[DeepSeek-V2](https://my.feishu.cn/wiki/PFQDwdWb4ivhtUkKTfPcBjXvnMd?from=from_copylink)中采用辅助损失来鼓励负载平衡，但过大的辅助损失会引入不可忽视的干扰梯度，从而损害模型性能。为了避免在训练过程中产生干扰梯度，V3 提出了一种无辅助损失的负载平衡策略（Loss-Free Balancing），通过动态调整每个专家的偏差来控制负载平衡，而不引入干扰梯度：

给每个专家一个偏置 $b_i$，只在路由（选 top-K）时把它加到亲和度（token-to-expert affinity）上：

$$
g_{i,t}' = \begin{cases} s_{i,t}, & s_{i,t} + b_i \in \operatorname{Topk}(\{s_{j,t} + b_j \mid 1 \leq j \leq N_r\}, K_r) \\ 0, & \text{otherwise} \end{cases}
$$

其中 $b_i$ 是第 $i$ 个专家的偏置。偏置只用于路由 Top-K 选择、不进入门控值（门控仍用原始亲和度 $s_{i,t}$）。

> **综合起来看无辅助损失负载均衡（Loss-Free Balancing）：**
>
> 令 $\mathbf{u}_t$ 是第 $t$ 个 token 的 FFN 输入，FFN 输出 $\mathbf{h}_t^{\prime}$ 为：
>
> $$
> \begin{aligned} & \mathbf{h}_t^{\prime}=\mathbf{u}_t+\sum_{i=1}^{N_s} \operatorname{FFN}_i^{(s)}\left(\mathbf{u}_t\right)+\sum_{i=1}^{N_r} g_{i, t} \operatorname{FFN}_i^{(r)}\left(\mathbf{u}_t\right), \\
> & g_{i, t}=\frac{g_{i, t}^{\prime}}{\sum_{j=1}^{N_r} g_{j, t}^{\prime}}, \\
> & g_{i, t}^{\prime}= \begin{cases}s_{i, t}, & s_{i, t}+b_i \in \operatorname{Topk}\left(\left\{s_{j, t}+b_j \mid 1 \leqslant j \leqslant N_r\right\}, K_r\right), \\ 0, & \text { otherwise, }\end{cases} \\
> & s_{i, t}=\operatorname{Sigmoid}\left(\mathbf{u}_t^{T} \mathbf{e}_i\right), \end{aligned}
> $$
>
> 其中 $N_s$ 和 $N_r$ 表示共享专家和路由专家的数量， $K_r$ 表示激活路由专家的数量， $g_{i,t}^{\prime}$ 是归一化前的门值， $g_{i,t}$ 是归一化后第 $i$ 个专家的混合权重， $\mathbf{e}_i$ 是当前层第 $i$ 个路由专家的中心（路由器权重矩阵的第 $i$ 行）， $s_{i,t}$ 是 token-to-expert affinity。
>
> 关键在于 $b_i$ 是专家 $i$ 的标量偏置，只参与 top-k 排序：被选中后混合权重取的仍是不含偏置的 $s_{i,t}$ ，而 top-k 本身不可导，所以 $b_i$ 既不进混合权重，也不给路由器引入任何额外梯度，推理时冻结。偏置每步结束后按反馈规则更新：
>
> $$
> b_i^{t+1} = b_i^{t} + \gamma\,\operatorname{sign}\!\left(\bar{\ell} - \ell_i^{t}\right)
> $$
>
> 其中 $\ell_i^t$ 是专家 $i$ 当前步实际接到的 token 数， $\bar{\ell} = \frac{1}{N_r}\sum_{j=1}^{N_r} \ell_j^t$ 是均值（即目标均衡负载）， $\gamma$ 是固定步长。直观理解为过载时偏置下调，欠载时偏置上调。这样即可用一个不进梯度的动态偏置把负载调均，避免了辅助损失的干扰梯度。

**互补序列级均衡损失：** 为了防止任何单个序列内的极端不平衡，我们还采用了互补的序列级辅助损失。这种损失鼓励每个序列中的专家负载保持平衡：

$$
\begin{aligned}
&\mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{N_r} f_i P_i,
\\ & f_i = \frac{N_r}{K_r T} \sum_{t=1}^{T} \mathbb{1}\big(s_{i,t} \in \operatorname{Topk}(\{s_{j,t}\}, K_r)\big),
\\ & s_{i,t}' = \frac{s_{i,t}}{\sum_{j=1}^{N_r} s_{j,t}},
\\ & P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t}'
\end{aligned}
$$

其中 $\alpha$ 是均衡因子（V3 取值极小）， $T$ 是序列 token 数， $\mathbb{1}(\cdot)$ 是指示函数， $f_i$ 是专家 $i$ 被选中的频率， $P_i$ 是归一化亲和度的平均。

**节点限制路由（Node-Limited Routing）：** 类比[DeepSeek-V2](https://my.feishu.cn/wiki/PFQDwdWb4ivhtUkKTfPcBjXvnMd?from=from_copylink)的设备受限路由，V3 用节点受限路由：每 token 至多发到 $M$ 个节点（按各节点上专家亲和度前 $\frac{K_r}{M}$ 之和选节点），使 MoE 训练几乎能完全重叠计算与通信。

**Token Dropping：** 因为负载均衡已经由以上技术保证，V3 训练与推理放弃 [DeepSeek-V2](https://my.feishu.cn/wiki/PFQDwdWb4ivhtUkKTfPcBjXvnMd?from=from_copylink) 的 token-dropping。

### 1.3 Multi-Token Prediction（MTP）

> 详细讲解参考：[MTP (Multi-Token Prediction) 原理+源码详解](https://my.feishu.cn/wiki/LH7Uwjg4BiTF5kklMtmcaDNqnyg?from=space_search)

**动机：** 本质就是串联多个 Head 预测多个 token，作用可以从训练侧和推理侧看：

- 训练侧：densify 训练监督信号。
- 推理侧：作为的草稿模型进行推测解码 （speculative decoding）、大幅减少 Decode 部分的访存压力。

#### 1.3.1 MTP 架构

![](https://cdn.nlark.com/yuque/0/2026/png/26585671/1793008973976-d802b445-90f5-4482-a13f-ca7cef0aacd6.png)

第 $k$ 个 MTP 模块含一个共享 embedding $\operatorname{Emb}(\cdot)$ 、一个共享输出头 $\operatorname{OutHead}(\cdot)$ 、一个 Transformer 块 $\operatorname{TRM}_k(\cdot)$ 和一个投影矩阵 $M_k \in \mathbb{R}^{d \times 2d}$ 。对第 $i$ 个 token、第 $k$ 个预测深度，先把第 $(k-1)$ 深度第 $i$ 个 token 的表示 $\mathbf{h}_i^{k-1} \in \mathbb{R}^d$ 与第 $(i+k)$ 个 token 的 embedding $\operatorname{Emb}(t_{i+k}) \in \mathbb{R}^d$ 拼接后线性投影：

$$
\mathbf{h}_i'^{k} = M_k [\operatorname{RMSNorm}(\mathbf{h}_i^{k-1}); \operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))]
$$

再送入该深度的 Transformer 块得到输出表示，最后由共享输出头计算预测概率：

$$
\begin{aligned}
&\mathbf{h}_{1:T-k}^{k} = \operatorname{TRM}_k(\mathbf{h}_{1:T-k}'^{k}),
\\ & P_{i+k+1}^{k} = \operatorname{OutHead}(\mathbf{h}_i^{k})
\end{aligned}
$$

- $d$ 是隐藏维度（embedding dimension）。
- $T$ 是输入序列长度， $i{:}j$ 表示切片（含左右边界）， $k = 1$ 时 $\mathbf{h}_i^{k-1}$ 就是主模型给出的表示。
- 每个 MTP 模块的 embedding 层与输出头都与主模型共享。

#### 1.3.2 MTP 训练目标

对每个预测深度算交叉熵损失，再对所有深度取平均、乘权重因子：

$$
\begin{aligned}
&\mathcal{L}_{\text{MTP}}^{k} = \operatorname{CrossEntropy}(P_{2+k:T+1}^{k}, t_{2+k:T+1}) = -\frac{1}{T} \sum_{i=2+k}^{T+1} \log P_i^{k}[t_i],
\\ & \mathcal{L}_{\text{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}_{\text{MTP}}^{k}
\end{aligned}
$$

其中 $t_i$ 是第 $i$ 位的 ground-truth token， $P_i^k[t_i]$ 是第 $k$ 个 MTP 模块给出的 $t_i$ 的预测概率， $D$ 是 MTP 模块数（预测深度）， $\lambda$ 是 MTP 损失权重。

#### 1.3.3 MTP 推理侧部署

在推理过程中，可以直接丢弃 MTP 模块，主模型可以独立且正常地工作，也可以将这些 MTP 模块重新用于推测性解码，以进一步改善生成延迟。

## 2. 训练 Infra

### 2.1 DualPipe：计算与通信重叠的流水线

**动机：** 跨节点专家并行（EP）带来的通信开销让计算通信比接近 1:1，若不重叠（Overlap）一半时间在等通信。

**实现：** DualPipe 把每对前向/反向 chunk 拆成四个组件（attention、all-to-all dispatch、MLP、all-to-all combine），反向 chunk 再把 attention 与 MLP 拆成两部分（backward for input、backward for weights），手动调 GPU SM 在通信与计算上的分配比例，使 **all-to-all 与 PP 通信被完全隐藏**。

![](https://cdn.nlark.com/yuque/0/2026/png/26585671/1793008993972-3a73667f-0afa-47f6-8631-0961d4a2ba55.png)

### 2.2 FP8 混合精度训练

**动机：** 低精度训练能省显存与算力，但 FP8 动态范围窄，激活/权重/梯度里的离群值会严重损失精度，此前很少有大模型训练用 FP8。

**实现：** V3 提出细粒度的 FP8 混合精度框架。

- **FP8 部分**：占算力主体的三个 GEMM（前向 Fprop、激活反向 Dgrad、权重反向 Wgrad）都用 FP8（E4M3）。
- **保持高精度**：embedding、输出头、MoE 门控、归一化、注意力算子仍用 BF16/FP32。master weight、权重梯度、优化器状态也存高精度（类似 Zero 摊薄显存开销）。

> Backward 时重算 RMSNorm 与 MLA up-projection（Recomputation）。EMA 参数放 CPU 异步更新。MTP 模块与主模型共享 embedding 与输出头、且部署同 PP rank 共享参数。

## 3. 预训练

14.8T token 高质量多语语料，训练序列长 4K，预训练最后阶段用 YaRN 两阶段扩到 128K。分词沿用 BBPE。训练全程用上文的 FP8 框架。总参 671B、激活 37B，训练成本极低（～278.8 万 H800 GPU 小时）。

## 4. 后训练

后训练两阶段 SFT + RL（GRPO）。

**SFT 阶段：** V3 的一个特点是把 DeepSeek-R1 的推理能力蒸馏进 V3（早期版本 R1）：用 R1 生成带验证与反思模式的推理数据做 SFT，把长链推理的能力迁移进来，同时控制输出风格与长度，避免 V3 输出过长。

**RL 阶段：** 沿用 GRPO 做对齐，相比[DeepSeek-V2](https://my.feishu.cn/wiki/PFQDwdWb4ivhtUkKTfPcBjXvnMd?from=from_copylink)，GRPO 阶段明确使用 Model-Based RM 和 Rule-Based RM（接近 R1 Setting）。
