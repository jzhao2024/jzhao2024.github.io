---
layout: post
title: Some Theoretical and Practical Thoughts on Diffusion Language Models（中文版）
date: 2026-08-08 10:00:00+0800
description: 过去一年 dLLM 工作的一次回顾，从 LLaDA-MoE 到 LLaDA 2.2，以及 dLLM 能否成为一条独立 scaling paradigm 的一些思考。
tags: dllm diffusion scaling
categories: notes
related_posts: false
toc:
  sidebar: left
---

**LLaDA Team, Ant Group**  
_Writing assisted by Codex/GPT5.6_

_本文的英文版见 [English version]({% post_url 2026-08-08-diffusion-language-models %})。_

过去一年，diffusion language model（dLLM）快速升温。Google 发布了 DiffusionGemma，LLaDA 系列推进到了 2.2，黄高老师团队的 [_The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models_](https://arxiv.org/abs/2601.15165) 也获得了 ICML Outstanding Paper。离散扩散语言模型已经从一个相对小众的研究方向，逐渐进入主流模型的讨论范围。

但一个始终没有消失的问题是：**dLLM 能取代 autoregressive language model（AR）吗？** 如果“取代”并不是一个恰当的问题，那么在未来的模型生态中，dLLM 又可能占据怎样的位置？

**本文并非一个严谨的tech report，更像是给对过去一年从LLaDA-Moe到上周发的LLaDA 2.2进行一次回顾，分享一些insights。**

非常随意，轻喷。

## 1. dLLM 正在进入主流视野

dLLM 的吸引力并不难理解。它改变了语言模型最基本的生成接口：模型不再只能从左向右逐个预测 token，而是可以在一个部分被遮盖的序列上反复去噪，同时更新多个位置。

这种生成方式带来了 full attention、灵活生成顺序、并行解码和反复修改等能力。随着模型规模、训练方法和推理系统逐渐成熟，dLLM 已经不再只是一个概念验证。今天真正值得讨论的，已经不是“diffusion 能不能生成语言”，而是它能否成为一条可以持续 scaling 的通用建模路线。

## 2. 先说明本文讨论的范围

本文主要讨论 masked、blockwise 的**离散扩散语言模型**，不包含 ELF 等连续扩散或 latent diffusion 方案。

这个边界很重要。目前许多有竞争力的 dLLM 都是从 AR checkpoint 改造而来，并在推理时采用 block diffusion。后文关于逐位置边际预测、并行提交和联合一致性的分析，主要针对这一类模型，不能不加区别地推广到所有 diffusion 方案。

连续 diffusion 是否具有相同的问题，仍然需要单独讨论。

## 3. 我们为什么会做 dLLM？

我们最初研究 dLLM，并不只是想替换一种解码器，而是希望在 next-token prediction 之外寻找一条 alternative scaling path。

AR 使用 causal attention，dLLM 可以使用 full attention；AR 固定从左到右生成，dLLM 可以在 block 内选择更灵活的生成顺序；AR 预测 next token，dLLM 则在原位置恢复 masked token；AR 通常逐 token 提交，dLLM 可以并行更新多个位置，并通过 remasking 对已有结果进行修改。

| AR                        | dLLM                         |
| ------------------------- | ---------------------------- |
| Causal attention          | Full/bidirectional attention |
| 固定 left-to-right order  | Block 内 any-order decoding  |
| Next-token prediction     | In-place denoising           |
| 逐 token 提交             | 多位置并行更新               |
| 已生成 token 通常不再修改 | 支持 remasking 和 revision   |

这些差异带来了一种很自然的期待：如果模型能够看到双向上下文，并在一次 forward 中处理多个位置，它是否可以绕开 AR 的顺序生成瓶颈？如果生成顺序不再固定，模型是否能够找到比 left-to-right 更高效的推理路径？如果模型能够 revision，它是否会更擅长全局规划和结构调整？

这些问题构成了 dLLM 最初的吸引力。

## 4. 从架构特性到 scaling 收益

但对于 scaling 来说，真正重要的问题并不是一种模型在形式上是否新颖，而是这些差异能否稳定转化成建模质量、数据效率和系统效率上的收益。

更具体地说，我们需要回答三个问题：

1. Full attention 是否带来了更有效的语言建模？
2. Any-order decoding 是否能够转化为可靠、可扩展的高速并行生成？
3. 为这些能力支付的训练、校准和系统成本，是否远超对应的 AR 方案？

这三个问题不能只通过比较少量 benchmark 或单次推理延迟来回答。它们最终需要放进完整的 scaling 过程：pretraining、post-training、serving 和持续迭代。

## 5. 第一性原理：一次并行去噪学到的是什么？

这是全文唯一需要稍微保留数学的部分，我们把它控制在三组公式以内。

### 5.1 dLLM 的训练目标

给定可见上下文 $$o$$ 和被遮盖的位置集合 $$M$$，模型逐位置优化 denoising loss。对每个位置而言，这个目标的最优解是该位置在当前上下文下的真实边际分布：

$$
\mathcal L
=
\mathbb E
\left[
-\sum_{i\in M}\log q_i(x_i\mid o)
\right],
\qquad
q_i^*(x_i\mid o)=p(x_i\mid o).
$$

也就是说，一次并行 readout 学到的是每个位置的边际分布。这个结论并不是在说模型容量不足；即使模型和优化都足够理想，逐位置交叉熵直接监督的仍然是每个位置各自的预测。

### 5.2 边际不等于联合

把同一次 forward 中多个位置的输出直接组合起来，相当于使用边际分布的乘积。但真正决定整段文本是否一致的是联合分布。两者之间缺失的依赖关系，可以用 total correlation 表示：

$$
\hat p(x_M\mid o)=\prod_{i\in M}p(x_i\mid o)
\quad\neq\quad
p(x_M\mid o),
\qquad
\mathrm{TC}(X_M\mid o)
=
\sum_i H(X_i\mid o)-H(X_M\mid o).
$$

用一句人话解释：

> **每个 token 单独看都很合理，不代表它们放在一起仍然合理。**

我们给一个形象的例子，用 `CLAUD` 和 `LLADA` 构造了一个很直观的例子。它考虑一个近似双峰的五槽位分布：概率质量主要集中在 `CLAUD`（“Claude”的前五个字母）和 `LLADA` 两个整体连贯的模式上，其他字符串的联合概率接近于零。加入很小的局部偏置后，我们给出的逐位置边际如下：

| 来源          |      Slot 1 |      Slot 2 |      Slot 3 |      Slot 4 |      Slot 5 |
| ------------- | ----------: | ----------: | ----------: | ----------: | ----------: |
| `CLAUD` mode  | **C: 0.52** | **L: 1.00** | **A: 1.00** |     U: 0.48 |     D: 0.48 |
| `LLADA` mode  |     L: 0.48 |     L: 1.00 |     A: 1.00 | **D: 0.52** | **A: 0.52** |
| 逐位置 argmax |       **C** |       **L** |       **A** |       **D** |       **A** |

于是，一个只看逐位置 argmax 的并行 decoder 会得到：

```text
CLAUD  ─┐
        ├─ per-position argmax → CLADA
LLADA  ─┘
```

问题在于，`CLADA` 既不是 `CLAUD`，也不是 `LLADA`。它从第一个模式中选走了开头的 `C`，又从第二个模式中选走了结尾的 `DA`。每个位置都遵循了自己的局部最优选择，拼出的字符串在真实联合分布中却几乎没有概率。

如果使用 AR 式的条件验证，这个混合很快就会暴露：一旦开头已经提交为 `C-L-A`，第四个位置更合理的条件预测应当是 `U`，而不是 `D`。一次并行 readout 看不到这个已经实现的条件关系，因为这五个位置是在同一个被遮盖的上下文中同时预测的。

这个例子并不是说 dLLM 最终一定会输出 `CLADA`，而是说明：**局部置信度无法单独为联合提交提供证明。** 当输出分布存在多个模式时，逐位置选择可能把不同模式的局部特征拼接到一起。

### 5.3 速度和联合一致性来自同一个旋钮

如果一次只提交一个 token，然后重新运行模型，后面的预测就可以条件于此前已经提交的结果，模型也可以重新获得位置之间的依赖。但这样做会增加 forward 次数，使生成逐渐接近 AR。

如果一次提交很多 token，模型可以获得更高的并行度，却更难只依靠局部置信度保证这些 token 联合一致。因此：

> **dLLM 的速度优势和它的联合一致性风险，本质上来自同一个变量：每次前向计算提交多少 token。**

我们再给出一个简单但是关键的充分条件。假设一次同时提交 $$k$$ 个 token，每个被选 token 的边际置信度都高于 $$\tau$$。在完全不知道这些 token 之间相关结构的最坏情况下，要保证这个组合至少具有非零联合概率，需要：

$$
\tau>1-\frac{1}{k}.
$$

$$k$$ 越大，这个阈值就越接近 1。比如同时提交 2 个 token 时阈值高于 0.5；提交 8 个时高于 0.875；提交 16 个时则高于 0.9375。这还只是保证组合的联合概率不为零，并不等价于保证它具有很高的正确概率。

这也解释了为什么并行提交最容易出现在格式、标点、括号、模板和复制片段等低熵区域：这些位置本来就接近确定。进入真正的语义分叉点后，模型要么减少同时提交的 token 数量，要么引入 remasking、search 或 verifier。

这里不再展开 Fréchet bound、synergy factor 等证明。更完整的数学推导、适用条件与边界情况，将在后续 technical report 中说明。

## Practical Thoughts

## 6. 从理论问题走向实际系统

### 6.1 实际 dLLM 为什么逐渐接近 AR？

理论问题很快会反映到系统设计中。为了提高训练稳定性和生成质量，当前较强的 dLLM 往往会采用：

- AR checkpoint 初始化；
- blockwise generation；
- 块间顺序生成；
- confidence-gated commitment；
- remasking、editing 或 verification；
- 较小的 block 来控制生成误差。

这些设计提升了训练和生成的可控性，同时也使实际系统逐渐呈现出 semi-autoregressive 的形态：block 内仍然可以并行或 any-order 解码，block 之间则保留顺序结构。

我们不倾向于把它简单描述为 dLLM “退化”成了 AR。一个可能更准确的理解是，AR 与 diffusion 并不是两个互斥的端点，而是一条连续谱。block size、commit policy 和 verifier 共同决定了系统在这条谱上的位置。

block 越大，并行空间越大，但联合提交越难；block 越小，训练和验证越可控，但系统也越接近 AR。

这还带来一个需要谨慎回答的问题：一个 AR-initialized dLLM 的能力，有多少来自 diffusion objective，又有多少来自已经由 AR pretraining 学到的语言表征？现有结果说明 diffusion 可以成为一种有效的生成和编辑接口，但尚不足以单独回答哪一种 objective 更适合作为通用的 scaling foundation。

### 6.2 dLLM 面对的并不只是 vanilla AR

AR 已经在大规模数据和模型上证明了自己的 scaling 能力。AR 与 dLLM 都存在训练—推理不一致：AR 有 exposure bias，dLLM 还需要处理 mask schedule、commit policy 和真实生成路径之间的差异。

但从目前的实践看，在长程生成、复杂推理和错误传播方面，AR 的行为总体上仍然更成熟，误差累积也更容易被观察和控制。

更重要的是，dLLM 今天的竞争对象并不是最原始的逐 token AR，而是：

> **AR foundation model + parallel decoding + verification**

[DFlash](https://arxiv.org/abs/2602.06036) 是一个直接的例子。它在固定的 AR target model 上训练轻量级 block-diffusion draft module，一次提出多个候选 token，再由 AR 模型进行验证。

[DSpark](https://arxiv.org/abs/2607.05147) 则使用 semi-autoregressive speculative module 和 confidence-scheduled verification，在保留 AR 验证机制的同时获得并行生成收益。

这些工作说明，parallel prediction 与 diffusion pretraining 并不必然绑定。基础语言建模可以继续依赖已经被充分验证的 AR scaling，再通过一个相对轻量的模块获得并行 proposal 能力，并把最终的联合一致性检查交给 AR verifier。

所以，dLLM 需要回答的问题已经不再只是“能否比逐 token AR 更快”，而是：

> 与 AR base model 加 speculative decoding 相比，完整 dLLM 能否在训练成本、生成质量、延迟和吞吐上提供更好的总体收益？

随着 speculative decoding、multi-token prediction 和 verifier-based systems 继续演进，这种竞争还会进一步加剧。dLLM 的优势最终需要在完整系统层面证明，而不能只比较单请求、低 batch 场景中的解码步数。

### 6.3 Any-order 也意味着 path-dependent

Any-order 是一种能力，但它不等于 order-invariant。

模型先提交哪些 token，会改变其余位置下一轮看到的条件。高置信度优先的策略通常有利于 pass@1：模型先填容易的位置，再利用这些结果处理困难的位置。但它也可能提前收窄后续搜索空间，使一些在早期仍然合理的生成路径不再被探索。

因此，生成顺序可能同时影响：

- 单次生成质量；
- 多样性和 exploration；
- 长程一致性；
- 最终生成分布；
- RL 中的 likelihood estimation。

从这个角度看，commit policy 并不只是一个实现细节。相同的 denoising model 配合不同的 mask schedule、threshold 或 remasking rule，可能产生不同的最终分布。

## 7. dLLM 可能更适合哪些位置？

上述分析也帮助我们理解 dLLM 更自然的应用位置。

Infilling、fill-in-the-middle、局部重写、结构化文本、代码补全和双向约束，往往包含大量可以从周围上下文确定的低熵区域，也允许模型反复修改。在这些任务中，bidirectional context 和 revision 不是附加能力，而可能直接成为建模优势。

相比之下，长链推理、数学证明、多步规划、agent tool use 和高事实性开放生成，需要状态随生成过程持续演化。一个早期决策会改变后续问题的条件，错误也可能沿着推理链持续传播。在这些场景中，当前 AR 仍然更加自然和成熟。

因此，dLLM 是否有价值与它能否全面替代 AR，是两个不同的问题。即使它不成为统一的基础模型范式，也可能在编辑、补全、结构生成和交互式建模中占据重要位置。

## 8. 真正值得寻找的，不只是更快的低熵生成

dLLM 已经说明，语言生成中确实存在大量可以并行处理的区域。标点、括号、缩进、Markdown 格式、模板、固定短语和复制片段，常常能在一次 forward 中被可靠地预测多个。

但这些区域同样可以被 AR speculative decoding 利用。如果 dLLM 的主要收益只是更快地生成本来就接近确定的 token，那么它与 AR 加速系统的差异可能更多在于成本被放在了哪里：dLLM 将更多成本放进 pretraining、adaptation 和 calibration，而 speculative decoding 将更多成本放在较小的 draft module 与 serving stack 中。

因此，对 dLLM 更有决定性的证据，可能不是低熵 continuation 上更高的 tokens per forward，而是它能否在以下问题上展现结构性优势：

- 全局编辑与结构性 revision；
- 双向约束满足；
- joint decoding；
- multi-path generation；
- AR 难以自然表达的新型交互方式。

要判断一种模型范式能否长期成立，最终还是需要回到 scaling。

## 9. 在 AGI 时代，核心问题仍然是 Scaling

我们可以用一个并不严格、但对实践有帮助的框架来理解 scaling：

$$
\text{Scaling}
\approx
(\text{Proper Modeling}+\text{Lots of Data})
\times
\text{Token Efficiency}
+
\text{Infrastructure Efficiency}.
$$

这不是一条数学定律，而是一种拆解问题的方式：一个模型范式需要能够高效吸收大量数据，并在有限的 token 和计算预算下，将这些数据稳定地转化成模型能力。

### 9.1 Modeling 与 lots of data

Transformer 与 LSTM 的故事已经不需要重复太多。Transformer 最重要的价值之一，是它能够有效吸收 LSTM 难以承载的数据和计算规模。架构的意义最终要通过 scaling 表现出来，而不只是来自形式上的优雅。

对于 dLLM 也是如此。Full attention、any-order 和 revision 都是有价值的 feature，但关键问题仍然是：在数据和算力持续增长时，它能否稳定转化出更强的模型能力？

数据量本身则相对直接。无论采用哪一种 objective，模型都需要可靠地吸收足够大规模、足够多样的数据。更容易被忽略的问题，是进入训练过程的每个 token 到底产生了多少有效监督。

### 9.2 Token efficiency

在我们进行的大量 block diffusion 实验中，mask ratio 通常位于 0.3–0.8，难以长期直接拉到 1。

未被 mask 的 token 仍然会作为上下文参与 attention 和计算，因此不能简单地说它们“没有被利用”；但它们在该轮不会提供直接的 denoising prediction loss。

相比之下，标准 AR pretraining 通常会对几乎所有目标位置计算 next-token loss。这意味着在相同语料 token 数下，block diffusion 每轮获得的显式监督密度可能天然低一些。对于真正的大规模训练，这个差异会直接影响 time-to-quality，而不仅仅是一项局部的 loss 设计选择。

Any-order 还带来另一层成本。为了减轻训练状态与实际生成路径之间的不一致，一个 block 往往不能只执行一次：

```text
mask → forward → loss → backward
```

而需要模拟多轮逐渐揭示 token 的过程。我们把这一类方法称为 MTF（multi-turn forward）。它可以让训练过程更接近真实的迭代生成，但也意味着同一个 block 需要更多次 forward。

MTF 的具体算法、收益和成本，我们会在后续 technical report 中详细说明。

类似地，以 LLaDA 2.2 中的 insert/delete 机制为例，更丰富的状态变化可以改善模型对生成路径的覆盖，却也需要额外的状态构造和 forward，进一步增加训练时长。

因此，评价 dLLM 的 token efficiency，不能只报告训练语料中包含多少 token，还需要一起考虑：

- 每轮提供直接监督的 token 比例；
- 每个 block 所需的 forward 次数；
- 为覆盖真实生成路径增加的计算；
- 达到相同模型质量所需的总训练成本。

### 9.3 Infrastructure efficiency

同样的数据、同样规模的模型和同样的集群，谁能更快、更稳定地收敛，也是 scaling 的一部分。这里包括 MFU、kernel 和通信效率、训练稳定性、单位 token 的 forward/backward 成本，以及最终的 time-to-quality。

后训练尤其值得关注。AR 的序列 log-probability 有一个直接的分解：

$$
\log p(x)=\sum_i\log p(x_i\mid x_{<i}).
$$

而 dLLM 的最终输出依赖 mask schedule、commit order、remasking 和 block transition。对于实际部署的 sampler，每个 token 通常没有一个像 AR next-token probability 那样直接、唯一且低成本可得的 log-probability。

问题不是概率在理论上不存在，而是它需要对大量潜在生成轨迹进行边缘化，或者通过近似方法估计。

如果算法最终需要回到 Monte Carlo trajectory estimation，为了控制估计方差，就可能需要更多 trajectory samples、更小的 microbatch，再来个额外的micromicrobatch，同时还要保存&回放更多中间状态。这些都会增加系统成本。

而 RL 从来都不是一个宽容的系统。早年，在Atari上，一个 random seed 就可能显著改变 policy；即使今天在 AR 上，数值精度、log-prob mismatch 或 batching 的细微差异，仍然可能冲击训练稳定性。把难以估计的 diffusion likelihood 再叠加到这个系统上，会进一步提高算法与 infrastructure 的复杂度。

我们为什么没做OPD？因为这里每一个token都有个估算不准的logp...

这并不意味着 dLLM 无法进行 RL，也不意味着 Monte Carlo 或 ELBO 类方法没有价值。更准确的结论是：相关方法的 reward gain，需要与 likelihood estimator 的偏差、方差、额外 forward 次数和系统复杂度一起报告。

## 10. 还有哪些问题仍然未知？

上述讨论主要描述了当前 dLLM 面临的问题，并不能决定它未来的上限。有几个方向尤其值得继续观察。

### 10.1 dLLM 能否成为 interaction model？

如果未来的模型不再是接收 prompt 后一次性输出长文本，而是在环境中持续观察、局部修改和反复交互，那么原位更新、双向条件、revision 和 arbitrary-order generation，可能会比传统静态文本生成更加自然。

沿着 Thinking Machines Lab 所讨论的 interaction model 方向，dLLM 会呈现怎样的**产品**形态，目前仍然未知。

### 10.2 连续 diffusion 会不会改变上述结论？

本文的边际—联合分析主要针对离散 masked readout。连续或 latent diffusion 可以采用不同的联合表示和全局向量场，可能绕开某些逐位置独立提交问题，也可能引入新的离散化、对齐和解码误差。

它需要一套独立的理论和系统分析。

### 10.3 具身

具身任务天然包含多模态状态、连续观察、局部动作修正、clean/noisy token 混合，以及非严格 left-to-right 的决策过程。事实上，不少工作已经在向这个方向靠近。

当 LingBot-VA 发布的时候，那个clean-noisy packing的attention mask是否看起来有点熟呢？

## 结语

dLLM 已经是一条值得继续投入的建模路线。它让 full attention、any-order generation、parallel decoding 和 revision 在语言模型中成为了可以被系统研究的能力，也提醒我们 next-token prediction 并不是唯一可能的生成接口。

但它能否成为一条独立的 scaling paradigm，仍取决于几个更朴素的指标：它是否能更有效地建模数据，是否具有足够高的 token efficiency，以及它是否能在训练、推理和后训练基础设施上保持可接受的成本和稳定性。

目前，这些问题都还没有最终答案。dLLM 可能不会以简单“取代 AR”的方式进入未来生态；更可能的情况是，AR、diffusion、speculative decoding、verification 和 interaction modeling 逐渐融合，而不同方法在各自更适合的任务中承担不同角色。

真正有趣的部分，可能才刚刚开始。

用丘吉尔的话说:

Now this is not the end. It is not even the beginning of the end.But it is, perhaps, the end of the beginning.

## References

1. _Discrete Diffusion Language Models Are Not Yet a Replacement for Autoregressive Language Models_.
2. Google. [DiffusionGemma Model Overview](https://ai.google.dev/gemma/docs/diffusiongemma).
3. Ni, Z. et al. [The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models](https://arxiv.org/abs/2601.15165), 2026. ICML Outstanding Paper.
4. Chen, J., Liang, Y., & Liu, Z. [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036), 2026.
5. Cheng, X. et al. [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147), 2026.
