---
layout: post
title: Some Theoretical and Practical Thoughts on Diffusion Language Models
date: 2026-08-08 10:00:00+0800
description: A retrospective on the past year of work on diffusion language models, from LLaDA-MoE to LLaDA 2.2, and on what it would take for dLLMs to become an independent scaling paradigm.
tags: dllm diffusion scaling
categories: notes
related_posts: false
toc:
  sidebar: left
_styles: >
  .post-content {
    font-size: 1.05rem;
    line-height: 1.8;
  }

  #markdown-content > p {
    margin-bottom: 1.3rem;
  }

  /* byline + language switch */

  #markdown-content > p:first-child {
    margin-bottom: 0.3rem;
    line-height: 1.6;
  }

  #markdown-content > p:first-child em {
    color: var(--global-text-color-light);
    font-size: 0.9rem;
  }

  #markdown-content > p:nth-child(2) {
    margin-bottom: 2.4rem;
    padding-bottom: 1.6rem;
    border-bottom: 1px solid var(--global-divider-color);
    font-size: 0.9rem;
  }

  /* headings */

  #markdown-content h2 {
    margin-top: 3.2rem;
    margin-bottom: 1.2rem;
    padding-bottom: 0.45rem;
    font-size: 1.55rem;
    font-weight: 600;
    line-height: 1.35;
    border-bottom: 1px solid var(--global-divider-color);
    scroll-margin-top: 5rem;
  }

  #markdown-content h3 {
    margin-top: 2.3rem;
    margin-bottom: 0.85rem;
    font-size: 1.28rem;
    font-weight: 600;
    scroll-margin-top: 5rem;
  }

  #markdown-content h2#practical-thoughts {
    margin: 4.5rem 0 2.2rem;
    padding: 1rem 0;
    border-top: 1px solid var(--global-divider-color);
    border-bottom: 1px solid var(--global-divider-color);
    text-align: center;
    font-size: 1.3rem;
    letter-spacing: 0.08em;
    color: var(--global-theme-color);
  }

  /* lists */

  #markdown-content ul,
  #markdown-content ol {
    margin-bottom: 1.4rem;
  }

  #markdown-content li {
    margin-bottom: 0.3rem;
  }

  /* tables */

  #markdown-content table {
    width: 100%;
    margin: 1.9rem 0;
    border-collapse: collapse;
    font-size: 0.95rem;
    line-height: 1.55;
  }

  #markdown-content table th {
    padding: 0.6rem 0.9rem;
    border-bottom: 2px solid var(--global-theme-color);
    font-weight: 600;
  }

  #markdown-content table td {
    padding: 0.55rem 0.9rem;
    border-top: 1px solid var(--global-divider-color);
    vertical-align: top;
  }

  #markdown-content table tbody tr:nth-child(even) {
    background-color: var(--global-code-bg-color);
  }

  /* pull quotes */

  #markdown-content blockquote {
    margin: 2rem 0;
    padding: 1.1rem 1.4rem;
    font-size: 1.05rem;
    line-height: 1.7;
    border-left: 4px solid var(--global-theme-color);
    background-color: var(--global-code-bg-color);
    border-radius: 0 6px 6px 0;
  }

  /* display equations */

  #markdown-content mjx-container[display="true"] {
    margin: 1.9rem -1.5rem;
    padding: 1.2rem 1.5rem;
    background-color: var(--global-code-bg-color);
    border-radius: 6px;
  }

  /* code blocks */

  #markdown-content div.highlight {
    margin: 1.6rem 0;
  }

  #markdown-content pre {
    padding: 0.9rem 1.1rem;
    line-height: 1.5;
    overflow-x: auto;
  }

  @media (max-width: 900px) {
    #markdown-content mjx-container[display="true"] {
      margin: 1.9rem 0;
      padding: 1.1rem 0.75rem;
      overflow-x: auto;
      overflow-y: hidden;
    }
  }

  @media (max-width: 600px) {
    #markdown-content table {
      display: block;
      overflow-x: auto;
      font-size: 0.85rem;
    }
    #markdown-content h2 {
      font-size: 1.35rem;
    }
  }
---

**LLaDA Team, Ant Group**  
_Writing assisted by Codex/GPT5.6_

_A Chinese version of this post is available [here]({% post_url 2026-08-08-diffusion-language-models-zh %})._

Over the past year, diffusion language models (dLLMs) have gained tremendous momentum. Google released DiffusionGemma, the LLaDA series advanced to version 2.2, and Professor Gao Huang's team's [_The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models_](https://arxiv.org/abs/2601.15165) received an ICML Outstanding Paper award. Discrete diffusion language models have moved from a relatively niche research direction into the mainstream conversation around language models.

Yet one persistent question remains: **Can dLLMs replace autoregressive language models (AR models)?** And if “replacement” is not quite the right framing, what role might dLLMs occupy in the future model ecosystem?

**This post is not meant to be a rigorous technical report. It is closer to a retrospective on our work over the past year—from LLaDA-MoE to LLaDA 2.2, which we released last week—together with a few insights we picked up along the way.**

This is a very informal post, so please take it lightly.

## 1. dLLMs Are Entering the Mainstream

The appeal of dLLMs is not difficult to understand. They change the most basic generation interface of a language model: instead of predicting tokens only from left to right, a model can repeatedly denoise a partially masked sequence and update multiple positions at once.

This generation process enables full attention, flexible generation order, parallel decoding, and iterative revision. As model scale, training methods, and inference systems continue to mature, dLLMs are no longer merely a proof of concept. The interesting question today is no longer whether diffusion can generate language, but whether it can become a general modeling approach that continues to scale.

## 2. Scope of This Discussion

This post focuses on **discrete diffusion language models** with masked, blockwise generation. It does not cover continuous or latent diffusion approaches such as ELF.

This distinction matters. Many competitive dLLMs today are adapted from AR checkpoints and use block diffusion at inference time. The analysis below—particularly the discussion of per-position marginals, parallel commitment, and joint consistency—primarily applies to this class of models. It should not be generalized indiscriminately to every diffusion approach.

Whether continuous diffusion faces the same issues requires a separate discussion.

## 3. Why Did We Work on dLLMs in the First Place?

When we began working on dLLMs, our goal was not merely to replace one decoder with another. We wanted to explore an alternative scaling path beyond next-token prediction.

AR models use causal attention, while dLLMs can use full attention. AR fixes generation to a left-to-right order, while dLLMs can choose a more flexible order within a block. AR predicts the next token, while a dLLM reconstructs masked tokens in place. AR typically commits one token at a time, while a dLLM can update multiple positions in parallel and revise previous predictions through remasking.

| AR                                       | dLLM                              |
| ---------------------------------------- | --------------------------------- |
| Causal attention                         | Full/bidirectional attention      |
| Fixed left-to-right order                | Any-order decoding within a block |
| Next-token prediction                    | In-place denoising                |
| Token-by-token commitment                | Parallel updates across positions |
| Generated tokens are usually not revised | Remasking and revision            |

These differences naturally create a set of expectations. If a model can see bidirectional context and process multiple positions in a single forward pass, can it bypass the sequential generation bottleneck of AR? If generation order is no longer fixed, can the model find a more efficient reasoning path than left-to-right decoding? If the model can revise its output, will it become better at global planning and structural adjustment?

These questions were a major part of the original appeal of dLLMs.

## 4. From Architectural Features to Scaling Benefits

For scaling, however, the important question is not whether a model is novel in form. It is whether those differences can be translated reliably into gains in modeling quality, data efficiency, and systems efficiency.

More concretely, we need to answer three questions:

1. Does full attention lead to more effective language modeling?
2. Can any-order decoding be converted into reliable, scalable, high-speed parallel generation?
3. Do the training, calibration, and systems costs of obtaining these capabilities far exceed those of the corresponding AR alternatives?

These questions cannot be answered by comparing a handful of benchmarks or single-request inference latency. They ultimately have to be evaluated across the complete scaling process: pretraining, post-training, serving, and continued iteration.

## 5. First Principles: What Does One Parallel Denoising Pass Learn?

This is the only part of the post where we retain a little mathematics, and we will keep it to three groups of equations.

### 5.1 The dLLM Training Objective

Given visible context $$o$$ and a set of masked positions $$M$$, the model optimizes a per-position denoising loss. At each position, the optimum of this objective is the true marginal distribution of that position under the current context:

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

In other words, a single parallel readout learns the marginal distribution at each position. This is not a statement about insufficient model capacity. Even with an ideal model and ideal optimization, the direct supervision provided by per-position cross-entropy still concerns each position's individual prediction.

### 5.2 Marginals Are Not the Joint

Combining the outputs of multiple positions from the same forward pass amounts to using a product of marginals. What actually determines whether the full span is coherent, however, is the joint distribution. The dependency information missing between the two can be expressed as total correlation:

$$
\hat p(x_M\mid o)=\prod_{i\in M}p(x_i\mid o)
\quad\neq\quad
p(x_M\mid o),
\qquad
\mathrm{TC}(X_M\mid o)
=
\sum_i H(X_i\mid o)-H(X_M\mid o).
$$

In plain language:

> **Every token can look reasonable on its own without the tokens being reasonable together.**

Table 2 of the paper gives an intuitive example using `CLAUD` and `LLADA`. It considers an approximately bimodal distribution over five slots, with most of the probability mass concentrated on two coherent modes: `CLAUD`—the first five letters of “Claude”—and `LLADA`. Other strings have close to zero joint probability. After adding small local biases, the paper gives the following per-position marginals:

| Source              |      Slot 1 |      Slot 2 |      Slot 3 |      Slot 4 |      Slot 5 |
| ------------------- | ----------: | ----------: | ----------: | ----------: | ----------: |
| `CLAUD` mode        | **C: 0.52** | **L: 1.00** | **A: 1.00** |     U: 0.48 |     D: 0.48 |
| `LLADA` mode        |     L: 0.48 |     L: 1.00 |     A: 1.00 | **D: 0.52** | **A: 0.52** |
| Per-position argmax |       **C** |       **L** |       **A** |       **D** |       **A** |

A parallel decoder that only follows the per-position argmax therefore produces:

```text
CLAUD  ─┐
        ├─ per-position argmax → CLADA
LLADA  ─┘
```

The problem is that `CLADA` is neither `CLAUD` nor `LLADA`. It takes the initial `C` from the first mode and the final `DA` from the second. Every position follows its own locally optimal choice, yet the assembled string has almost no probability under the true joint distribution.

An AR-style conditional verification would expose this hybrid almost immediately. Once the prefix `C-L-A` has been committed, the more plausible conditional prediction at the fourth position is `U`, not `D`. A single parallel readout cannot see this realized conditional relationship because all five positions were predicted simultaneously under the same masked context.

This example does not claim that a dLLM must ultimately output `CLADA`. It shows that **local confidence alone cannot certify a joint commitment**. When the output distribution has multiple modes, independent position-wise selection can splice local features from different modes into one sequence.

### 5.3 Speed and Joint Consistency Share the Same Knob

If the model commits only one token at a time and then runs again, later predictions can condition on the tokens already committed. The model can recover dependencies across positions, but at the cost of additional forward passes, making generation progressively more AR-like.

If the model commits many tokens at once, it gains more parallelism, but local confidence alone becomes less able to guarantee that those tokens are jointly consistent. Therefore:

> **The speed advantage of a dLLM and its risk of joint inconsistency fundamentally come from the same variable: how many tokens it commits per forward pass.**

The paper also gives a simple sufficient condition. Suppose the model commits $$k$$ tokens at once and the marginal confidence of every selected token exceeds $$\tau$$. In the worst case, with no knowledge of the correlation structure among those tokens, guaranteeing that the combination has at least nonzero joint probability requires:

$$
\tau>1-\frac{1}{k}.
$$

As $$k$$ increases, this threshold approaches 1. For two simultaneously committed tokens, the threshold is above 0.5; for eight, it is above 0.875; and for sixteen, it is above 0.9375. Even this guarantees only that the joint probability is nonzero—not that the combination has a high probability of being correct.

This also explains why parallel commitment is easiest in low-entropy regions such as formatting, punctuation, brackets, templates, and copied spans: these positions are already close to deterministic. At genuine semantic branch points, the model must either reduce the number of tokens committed together or introduce remasking, search, or a verifier.

We will not expand on the Fréchet bound, the synergy factor, or related proofs here. A future technical report will cover the full derivation, its assumptions, and its boundary cases.

## Practical Thoughts

## 6. From the Theoretical Issue to Real Systems

### 6.1 Why Are Practical dLLMs Becoming More AR-Like?

The theoretical issue quickly appears in system design. To improve training stability and generation quality, stronger dLLMs today commonly use:

- AR checkpoint initialization;
- blockwise generation;
- sequential generation across blocks;
- confidence-gated commitment;
- remasking, editing, or verification;
- smaller blocks to control generation errors.

These choices make training and generation more controllable, while also giving practical systems an increasingly semi-autoregressive form: decoding can remain parallel or any-order within each block, but the blocks themselves preserve a sequential structure.

We do not think it is helpful to describe this simply as dLLMs “degenerating” into AR. A more accurate view may be that AR and diffusion are not two mutually exclusive endpoints, but positions on a continuum. Block size, commit policy, and the verifier jointly determine where a system lies on that continuum.

Larger blocks create more room for parallelism, but make joint commitment harder. Smaller blocks are easier to train, calibrate, and verify, but move the system closer to AR.

This also raises a question that requires careful interpretation: in an AR-initialized dLLM, how much capability comes from the diffusion objective, and how much comes from language representations already learned through AR pretraining? Existing results show that diffusion can be an effective generation and editing interface, but they do not yet establish which objective is the better general foundation for scaling.

### 6.2 A dLLM Is Not Competing Only with Vanilla AR

AR has already demonstrated its ability to scale across large models and large datasets. Both AR and dLLMs face some degree of train-inference mismatch: AR has exposure bias, while dLLMs must additionally handle differences among the mask schedule, commit policy, and actual generation trajectory.

In current practice, however, AR behavior remains more mature in long-horizon generation, complex reasoning, and error propagation. Its accumulated errors are also easier to observe and control.

More importantly, the competitor facing a dLLM today is not primitive token-by-token AR. It is:

> **AR foundation model + parallel decoding + verification**

[DFlash](https://arxiv.org/abs/2602.06036) is a direct example. It trains a lightweight block-diffusion draft module on top of a fixed AR target model, proposes multiple candidate tokens at once, and then lets the AR model verify them.

[DSpark](https://arxiv.org/abs/2607.05147) uses a semi-autoregressive speculative module with confidence-scheduled verification, gaining parallel generation while retaining AR verification.

These systems show that parallel prediction does not have to be coupled with diffusion pretraining. Foundation-model training can continue to rely on the well-tested scaling behavior of AR, while a relatively lightweight module supplies parallel proposals and an AR verifier checks their joint consistency.

The question a dLLM must answer is therefore no longer merely “Can I run faster than token-by-token AR?” It is:

> Compared with an AR base model plus speculative decoding, can a full dLLM deliver a better overall tradeoff across training cost, generation quality, latency, and throughput?

As speculative decoding, multi-token prediction, and verifier-based systems continue to improve, this competition will become even more intense. The benefits of dLLMs ultimately have to be demonstrated at the level of the complete system, not merely through decoding-step counts for a single request at low batch size.

### 6.3 Any-Order Also Means Path-Dependent

Any-order is a capability, but it does not imply order invariance.

The tokens committed first change the conditions seen by all remaining positions in subsequent rounds. A high-confidence-first policy is often good for pass@1: the model fills easy positions first, then uses them as context for harder positions. But it can also narrow the search space too early, leaving some trajectories that looked plausible in an earlier state unexplored.

Generation order can therefore affect:

- single-sample quality;
- diversity and exploration;
- long-range consistency;
- the final generation distribution;
- likelihood estimation in RL.

From this perspective, the commit policy is not just an implementation detail. The same denoising model paired with a different mask schedule, threshold, or remasking rule may induce a different final distribution.

## 7. Where Might dLLMs Fit Best?

The analysis above also helps clarify the applications to which dLLMs are more naturally suited.

Infilling, fill-in-the-middle, local rewriting, structured text, code completion, and bidirectional constraint satisfaction often contain many low-entropy regions that can be recovered from surrounding context. They also allow the model to revise its output. In these tasks, bidirectional context and revision are not merely additional features; they may directly become modeling advantages.

By contrast, long chains of reasoning, mathematical proofs, multi-step planning, agentic tool use, and high-factuality open-ended generation require the state to evolve continuously throughout generation. An early decision changes the conditions of the subsequent problem, and an error can propagate along the entire reasoning chain. In these settings, current AR systems remain more natural and mature.

Whether dLLMs are valuable and whether they can fully replace AR are therefore two different questions. Even if dLLMs do not become a unified foundation-model paradigm, they may still occupy an important role in editing, infilling, structured generation, and interactive modeling.

## 8. What We Should Look For Is More Than Faster Low-Entropy Generation

dLLMs have already shown that language generation contains many regions that can be processed in parallel. Punctuation, brackets, indentation, Markdown formatting, templates, fixed phrases, and copied spans can often be predicted several tokens at a time in a single forward pass.

But AR speculative decoding can exploit the same regions. If the main benefit of a dLLM is faster generation of tokens that were already nearly deterministic, then the difference between dLLMs and AR acceleration systems may primarily be where the cost is paid. A dLLM places more of the cost in pretraining, adaptation, and calibration, while speculative decoding places more of it in a smaller draft module and the serving stack.

More decisive evidence for dLLMs may therefore come not from higher tokens per forward pass on low-entropy continuations, but from structural advantages in:

- global editing and structural revision;
- bidirectional constraint satisfaction;
- joint decoding;
- multi-path generation;
- new forms of interaction that are difficult to express naturally with AR.

To decide whether a modeling paradigm can endure, however, we eventually have to return to scaling.

## 9. In the Age of AGI, the Core Problem Is Still Scaling

We can use a deliberately informal but practically useful framework to think about scaling:

$$
\text{Scaling}
\approx
(\text{Proper Modeling}+\text{Lots of Data})
\times
\text{Token Efficiency}
+
\text{Infrastructure Efficiency}.
$$

This is not a mathematical law. It is a way of decomposing the problem: a modeling paradigm must absorb large amounts of data efficiently and convert that data into model capability under finite token and compute budgets.

### 9.1 Modeling and Lots of Data

There is little need to retell the story of Transformers and LSTMs. One of the most important advantages of the Transformer is that it can absorb scales of data and compute that LSTMs could not. The value of an architecture must ultimately show up in scaling, not merely in the elegance of its formulation.

The same standard should apply to dLLMs. Full attention, any-order generation, and revision are all valuable features. But the key question remains: as data and compute continue to grow, can those features be converted reliably into stronger model capability?

The need for large amounts of data is straightforward. Regardless of the objective, a model must reliably absorb sufficiently large and diverse datasets. The less obvious question is how much useful supervision each token entering the training process actually provides.

### 9.2 Token Efficiency

Across a large number of our block-diffusion experiments, the mask ratio generally falls between 0.3 and 0.8 and is difficult to keep at 1 for sustained training.

Unmasked tokens still participate in attention and computation as context, so it would be inaccurate to say that they are “unused.” But they do not provide a direct denoising prediction loss in that training round.

By comparison, standard AR pretraining usually applies a next-token loss to almost every target position. For the same number of corpus tokens, block diffusion may therefore receive intrinsically sparser explicit supervision in each round. At true pretraining scale, this difference directly affects time-to-quality; it is not merely a local choice of loss design.

Any-order generation introduces another layer of cost. To reduce mismatch between training states and actual generation trajectories, a block often cannot be trained with only one round of:

```text
mask → forward → loss → backward
```

Instead, training must simulate multiple rounds in which tokens are gradually revealed. We refer to this family of methods as MTF, or multi-turn forward. It can bring training closer to the actual iterative generation process, but it also means that the same block requires more forward passes.

We will discuss the specific algorithm, benefits, and costs of MTF in a future technical report.

Similarly, mechanisms such as insert/delete in LLaDA 2.2 can improve coverage over generation trajectories through richer state transitions, but they also require additional state construction and forward passes, further increasing training time.

Token efficiency for dLLMs therefore cannot be evaluated only by counting corpus tokens. We also need to consider:

- the fraction of tokens providing direct supervision in each round;
- the number of forward passes required per block;
- the additional computation needed to cover realistic generation trajectories;
- the total training cost required to reach the same model quality.

### 9.3 Infrastructure Efficiency

Given the same data, the same model scale, and the same cluster, which system converges faster and more reliably is also part of scaling. This includes MFU, kernel and communication efficiency, training stability, forward/backward cost per token, and ultimately time-to-quality.

Post-training is especially important. For an AR model, sequence log probability has a direct decomposition:

$$
\log p(x)=\sum_i\log p(x_i\mid x_{<i}).
$$

The final output of a dLLM, by contrast, depends on the mask schedule, commit order, remasking, and block transitions. Under the deployed sampler, each token typically does not have a log probability that is as direct, unique, and inexpensive to obtain as an AR next-token probability.

The issue is not that such a probability does not exist in theory. It is that computing it requires marginalizing over a large number of latent generation trajectories or estimating it with an approximation.

If the algorithm eventually falls back to Monte Carlo trajectory estimation, controlling estimator variance may require more trajectory samples, a smaller microbatch, and then an additional micro-microbatch on top of that—all while storing and replaying more intermediate states. Each of these raises the systems cost.

RL has never been a forgiving system. In the early days on Atari, changing a single random seed could materially alter the learned policy. Even today, with AR models, small differences in numerical precision, log-probability mismatch, or batching can still destabilize training. Adding a difficult-to-estimate diffusion likelihood on top of that further increases both algorithmic and infrastructure complexity.

Why did we not do OPD? Because here every token comes with a log probability that is not estimated quite accurately...

This does not mean that RL for dLLMs is impossible, nor that Monte Carlo or ELBO-based methods have no value. A more accurate conclusion is that any reported reward gain should be presented together with the bias and variance of the likelihood estimator, the number of additional forward passes, and the resulting systems complexity.

## 10. What Is Still Unknown?

The discussion above describes challenges faced by current dLLMs; it does not determine their ultimate ceiling. Several directions are particularly worth watching.

### 10.1 Can dLLMs Become Interaction Models?

If future models no longer take a prompt and emit one long static response, but instead continuously observe an environment, make local modifications, and interact over time, then in-place updates, bidirectional conditioning, revision, and arbitrary-order generation may become more natural than conventional static text generation.

Along the interaction-model direction discussed by Thinking Machines Lab, what kind of **product** might a dLLM become? We do not yet know.

### 10.2 Will Continuous Diffusion Change the Picture?

The marginal-versus-joint analysis in this post primarily concerns discrete masked readouts. Continuous or latent diffusion may use different joint representations and global vector fields, potentially avoiding some forms of independent per-position commitment while introducing new discretization, alignment, and decoding errors.

It requires a separate theoretical and systems analysis.

### 10.3 Embodiment

Embodied tasks naturally involve multimodal state, continuous observation, local action correction, mixtures of clean and noisy tokens, and decision processes that are not strictly left-to-right. In fact, many projects are already moving in this direction.

When LingBot-VA was released, did its clean-noisy packed attention mask look a little familiar?

## Conclusion

dLLMs have already become a modeling direction worth continued investment. They have made full attention, any-order generation, parallel decoding, and revision into capabilities that can be studied systematically in language models. They also remind us that next-token prediction is not the only possible generation interface.

Whether dLLMs can become an independent scaling paradigm, however, still depends on a few more basic criteria: whether they model data more effectively, whether they achieve sufficiently high token efficiency, and whether their training, inference, and post-training infrastructure can remain acceptably efficient and stable.

None of these questions has a final answer yet. dLLMs may not enter the future ecosystem by simply “replacing AR.” A more likely outcome is that AR, diffusion, speculative decoding, verification, and interaction modeling gradually converge, with different approaches taking on different roles in the tasks to which they are best suited.

The truly interesting part may only be beginning.

In Churchill's words:

> Now this is not the end. It is not even the beginning of the end. But it is, perhaps, the end of the beginning.

## References

1. _Discrete Diffusion Language Models Are Not Yet a Replacement for Autoregressive Language Models_.
2. Google. [DiffusionGemma Model Overview](https://ai.google.dev/gemma/docs/diffusiongemma).
3. Ni, Z. et al. [The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models](https://arxiv.org/abs/2601.15165), 2026. ICML Outstanding Paper.
4. Chen, J., Liang, Y., & Liu, Z. [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036), 2026.
5. Cheng, X. et al. [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147), 2026.
