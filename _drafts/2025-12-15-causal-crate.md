---
layout: distill
title: Causal CRATE
description: 
tags: causal-crate
giscus_comments: true
date: 2025-12-15
featured: false
published: false
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: Ng Ka Lok
    url: "ngkel.github.io"
    affiliations:
      name: Ex-HKU, Foodpanda

bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Causal Sequence Processing
  - name: Definition of Causal Encoder
  - name: Constructing Causal Architectures
  - name: Efficient Inference with Caching
  - name: Training Causal Models

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
pseudocode: true
---

# Causal Sequence Processing

Some of the most popular applications of deep networks are in the regimes of \textit{sequence data} where, unlike the previous example of imagery, there is a natural order to the data. For example, unlike in imagery (where one tends to look at the whole image at once), language is processed one token at a time (causally). This simple fact motivates the development of large language models (LLMs), which are trained to predict the (probability distribution of) the next token in a sequence given \textit{only the history of previous tokens}. Then, to sample from an LLM, one simply iteratively samples from the predicted distribution for the next token and then appends the sampled token to the history. In order to train LLMs, and other models such as video generation models, it is necessary to develop architectures which can efficiently perform this causal computation.

# Definition of Causal Encoder

With this in mind, and the understanding of causal autoregressive processes from \Cref{ch:intro}, we say that an encoder \(f\) is \textit{causal} if for every input \(\vX = [\vx_{1}, \dots, \vx_{N}]\), it holds 

$$\begin{align}
    f(\vX)_{1:N-1} = f([\vx_{1}, \dots, \vx_{N}])_{1:N-1} = f([\vx_{1}, \dots, \vx_{N - 1}]) = f(\vX_{1:N-1}),
\end{align}$$
where \(\vZ_{1:N-1} = [\vz_{1}, \dots, \vz_{N-1}]\), etc. In English, this means that the first \(N-1\) features are equal to the output of the encoder on the first \(N-1\) token embeddings of the input; even simpler, it means we can compute the features of each token \textit{one at a time}, and this would be equivalent to computing the features for the entire input. This causality is necessary for the previously mentioned applications (e.g., LLMs). In particular, the previously presented implementations of CRATE and ToST are \textit{not causal} (proof is left as an exercise).

# Constructing Causal Architectures

We can construct causal white-box architectures by a variety of methods. Here, we will showcase a simple method which builds on our previous unrolled optimization framework. Specifically, we compute the features \(\vz_{1}, \dots, \vz_{N}\) corresponding to the first \(N\) tokens of the input \textit{one-at-a-time} to optimize the representation learning objective, such as the sparse rate reduction \eqref{eq:sparse-rr-1}:

$$\begin{align}
    \max_{\vz_{i}}[\Delta R(\vZ_{1:i} \mid \vU_{[K]}) - \lambda \|\vZ_{1:i}\|_{1}],\quad \forall i \in [N].
\end{align}$$

If we follow through with the two-step unrolling procedure that yielded CRATE, we can obtain the iteration:

$$\begin{align}
    \vz_{i}^{\ell + 1/2} &\approx \vz_{i}^{\ell} - \kappa \nabla_{\vz_{i}} R(\vZ_{1:i}^{\ell} \mid \vU_{[K]}^{\ell}), \\ 
    \vz_{i}^{\ell + 1} &\approx \argmin_{\vz_{i}} \left\{\lambda \|\vz_{i}\|_{1} + \frac{1}{2}\norm{\vz_{i}^{\ell + 1/2} - \vD^{\ell} {\vz_{i}}}_F^2\right\},
\end{align}$$

or, using the same conversion from these two steps to network operators that we used for CRATE,

$$\begin{align}
    \vz_{i}^{\ell + 1/2}
    &= \left(1 - \frac{\kappa p}{N\epsilon^{2}}\right)\vz_{i}^{\ell} + \frac{\kappa p}{N \epsilon^{2}}\operatorname{MSSA}(\vZ_{1:i}^{\ell} \mid \vU_{[K]}^{\ell})_{i}, \\ 
    \vz_{i}^{\ell + 1}
    &= \operatorname{ISTA}(\vz_{i}^{\ell + 1/2} \mid \vD^{\ell}).
\end{align}$$
# Efficient Inference with Caching

Let us investigate this iteration in slightly more detail. First, let us note that by construction, this sequence of features corresponds to a causal encoder. Next, let us suppose that we are in a setting where we have computed \(\vZ_{1:i-1}^{\ell + 1} = [\vz_{1}^{\ell + 1}, \dots, \vz_{i-1}^{\ell + 1}]\), having computed the quantities \((\vU_{k}^{\ell})^{\top}\vZ_{1:i-1}^{\ell} = [(\vU_{k}^{\ell})^{\top}\vz_{1}^{\ell}, \dots, (\vU_{k}^{\ell})^{\top}\vz_{i-1}^{\ell}]\) along the way, and want to compute \(\vz_{i}^{\ell + 1}\) (for instance relevant to the case of LLM inference). In this case, note that the update rule for \(\vz_{i}^{\ell + 1/2}\) can be simplified as:

$$\begin{align}
    \vz_{i}^{\ell + 1/2}
    &= \left(1 - \frac{\kappa p}{N\epsilon^{2}}\right)\vz_{i}^{\ell} + \frac{\kappa p}{N \epsilon^{2}}\operatorname{MSSA}(\vZ_{1:i}^{\ell} \mid \vU_{[K]}^{\ell})_{i} \\ 
    &= \left(1 - \frac{\kappa p}{N\epsilon^{2}}\right)\vz_{i}^{\ell} + \frac{\kappa p^{2}}{N^{2} \epsilon^{4}}\sum_{k =  1}^{K}\vU_{k}\vU_{k}^{\top}\vZ_{1:i}\softmax((\vU_{k}^{\top}\vZ_{1:i})^{\top}(\vU_{k}^{\top}\vZ_{1:i}))\ve_{i} \\
    &= \left(1 - \frac{\kappa p}{N\epsilon^{2}}\right)\vz_{i}^{\ell} + \frac{\kappa p^{2}}{N^{2} \epsilon^{4}}\sum_{k =  1}^{K}\vU_{k}\vU_{k}^{\top}\vZ_{1:i}\softmax((\vU_{k}^{\top}\vZ_{1:i})^{\top}(\vU_{k}^{\top}\vz_{i})) \\
    &= \left(1 - \frac{\kappa p}{N\epsilon^{2}}\right)\vz_{i}^{\ell} + \frac{\kappa p^{2}}{N^{2} \epsilon^{4}}\sum_{k =  1}^{K}\vU_{k}\vU_{k}^{\top}[\vZ_{1:i-1}, \vz_{i}]\softmax((\vU_{k}^{\top}[\vZ_{1:i-1}, \vz_{i}])^{\top}(\vU_{k}^{\top}\vz_{i})) \\
    &= \left(1 - \frac{\kappa p}{N\epsilon^{2}}\right)\vz_{i}^{\ell} + \frac{\kappa p^{2}}{N^{2} \epsilon^{4}}\sum_{k =  1}^{K}\vU_{k}[\vU_{k}^{\top}\vZ_{1:i-1}, \vU_{k}^{\top}\vz_{i}]\softmax(([\vU_{k}^{\top}\vZ_{1:i-1}, \vU_{k}^{\top}\vz_{i}])^{\top}(\vU_{k}^{\top}\vz_{i})).
\end{align}$$ 
This step now becomes \textit{highly efficient} if we cache \(\vU_{k}^{\top}\vZ_{1:i-1}\) from previous steps, and add a single column to it each time we compute this operator. Namely, we greatly reduce the number of large matrix-matrix products and replace them by cache loads and matrix-vector products, overall much cheaper in terms of time complexity. This caching is the reason why causal generative models such as LLMs can efficiently sample 1000s of tokens per second, even if each training step takes a few seconds by itself. The cache for the subspace projections of the features is also known as the so-called ``KV cache''.

# Training Causal Models

Finally, let us consider the case where we are to \textit{train} a causal CRATE model, and want to find the most efficient way to do this given a full input sequence \(\vX = [\vx_{1}, \dots, \vx_{N}]\) simultaneously. The ISTA step is parallelizable to become the regular ISTA step from the non-causal CRATE, i.e.,

$$\begin{align}
    \operatorname{ISTA}(\vZ \mid \vD) \doteq [\ISTA(\vz_{1} \mid \vD), \dots, \ISTA(\vz_{N} \mid \vD)],
\end{align}$$

therefore implying that the ISTA step remains the same as the non-causal CRATE. The MSSA step is more interesting, since it changes in a meaningful way. To see how it changes, note that each MSSA operator has an interesting structure which merits some attention:

$$\begin{align}
    \operatorname{MSSA}(\vz_{1:i} \mid \vU_{[K]})_{i} 
    &= \frac{p}{N\epsilon^{2}}\sum_{k =  1}^{K}\vU_{k}\vU_{k}^{\top}\vZ_{1:i}\softmax((\vU_{k}^{\top}\vZ_{1:i})^{\top}(\vU_{k}^{\top}\vZ_{1:i}))\ve_{i} \\
    &= \frac{p}{N\epsilon^{2}}\sum_{k =  1}^{K}\vU_{k}\vU_{k}^{\top}\vZ_{1:i}\softmax((\vU_{k}^{\top}\vZ_{1:N})^{\top}(\vU_{k}^{\top}\vz_{i})) \\ 
    &= \frac{p}{N\epsilon^{2}}\mat{\vU_{1}, \dots, \vU_{K}}\mat{\softmax((\vU_{1}^{\top}\vZ_{1:i})^{\top}(\vU_{1}^{\top}\vz_{i})) \\ \vdots \\ \softmax((\vU_{K}^{\top}\vZ_{1:i})^{\top}(\vU_{K}^{\top}\vz_{i}))}
\end{align}$$

Namely, if we define the \textit{causal MSSA operator} as the block matrix:

$$\begin{align}
    \operatorname{CausalMSSA}(\vZ \mid \vU_{[K]}) \doteq [\operatorname{MSSA}(\vZ_{1:1} \mid \vU_{[K]})_{1}, \dots, \operatorname{MSSA}(\vZ_{1:N} \mid \vU_{[K]})_{N}],
\end{align}$$

then, working our way through the softmax algebra (proof is again left as an exercise), we can show that

$$\begin{align}
    \operatorname{CausalMSSA}(\vZ \mid \vU_{[K]}) 
    &\doteq \frac{p}{N\epsilon^{2}}\mat{\vU_{1}, \dots, \vU_{K}}\mat{\softmax((\vU_{1}^{\top}\vZ)^{\top}(\vU_{1}^{\top}\vZ) +\vM_{N}) \\ \vdots \\ \softmax((\vU_{K}^{\top}\vZ)^{\top}(\vU_{K}^{\top}\vZ) + \vM_{N})}, \\ 
    \text{where} \quad &\vM_{N} = \mat{0 & -\infty & -\infty & \dots & -\infty & -\infty \\ 0 & 0 & -\infty & \dots & -\infty & -\infty \\ \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & \dots & 0 & -\infty \\ 0 & 0 & 0 & \dots & 0 & 0} \in \R^{N \times N}.
\end{align}$$
Here the matrix \(\vM_{N}\) is another way to encode that no feature in CausalMSSA depends on a future feature, since the relevant entries of the argument to the softmax are \(-\infty\), and thus are set to \(0\) after the exponential within the softmax, and therefore effectively ignored. This matrix \(\vM_{N}\) is sometimes called a \textit{(causal) attention mask}.

The upshot of this is that when we have the full sequence \(\vZ^{\ell}\), we can compute \(\vZ^{\ell + 1}\) in time similar to or less than that of the usual CRATE layer (since \(\vM_{N}\) is hard-coded, input-independent, and enables us to ignore many entries for the softmax). Thus, we can define and train the full sequence-to-sequence model in the same way as the regular architecture.

When we apply this to language modeling (details to be provided in \Cref{ch:applications}), we find that we can obtain reasonable results compared to similar-sized empirically designed language models, as shown in \Cref{tab:causal-crate-results}.
