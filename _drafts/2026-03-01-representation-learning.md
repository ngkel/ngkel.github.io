---
layout: distill
title: representation learning
description: 
tags: representation_learning
giscus_comments: true
date: 2025-08-18
featured: true
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
  - name: What learning is all about? Or at least, what are modern methods actually doing?
  - name: Learning problems that are solvable by analytical approach
  # if a section has subsections, you can add them as follows:
    subsections:
      - name: PCA - finding the single subspace that best fits the data
      - name: Power Iteration - Workhorse of PCA
      - name: Limitation of PCA
      - name: Mixtures of Subspaces and Sparsely-Used Dictionaries
      - name: Overcomplete dictionary learning
  - name: Learned ISTA

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
---
# Why do we want structured form of data distribution?
Recall that we can access arbitrary distribution with low dimensional support via iterative denoising. By Tweedie's formula, we connect the gradient of distribution's log-density with denoiser, thus making learning a denoiser a proxy of learning the data distribution. The problem is, for general distribution that is not as structured as GMM, we have no explicit functional form for the denoiser. For GMM, the functional form can be written, for example, 

$$\begin{align}
\bar{\mathbf{x}}^*(t, \mathbf{x}_t) = \sum_{k=1}^K \frac{\varphi(\mathbf{x}_t; \mathbf{0}, \mathbf{U}_k \mathbf{U}_k^\top + t^2 \mathbf{I})}{\sum_{i=1}^K \varphi(\mathbf{x}_t; \mathbf{0}, \mathbf{U}_i \mathbf{U}_i^\top + t^2 \mathbf{I})} \cdot (\mathbf{U}_k \mathbf{U}_k^\top (\mathbf{U}_k \mathbf{U}_k^\top + t^2 \mathbf{I})^{-1} \mathbf{x}_t)
\end{align}$$

Without standard structured form of data distribution that facilitates efficient access, it becomes difficult to use it for subsequent tasks, such as treating the distribution as prior for Bayesian inference. 

# How structured representation provide explicit functional form for conditional inference

Given a standard structured distribution such as GMM,

$$
x \sim \frac{1}{K}\sum_{k=1}^{K} \mathcal{N}(0, U_k U_k^{\top})
$$

The conditional sampling given signal $y$ can be written as:

$$
\bar{\boldsymbol{x}}_\theta^{\text{CFG}}(t, \boldsymbol{x}_t, y)
  = (1-\gamma)\,\bar{\boldsymbol{x}}_\theta(t, \boldsymbol{x}_t, \varnothing)
  + \gamma\,\bar{\boldsymbol{x}}_\theta(t, \boldsymbol{x}_t, y)
$$

where

$$\,\bar{\boldsymbol{x}}_\theta(t, \boldsymbol{x}_t, \varnothing) = \frac{1}{1+t^2}
\sum_{k=1}^{K}
\frac{
  \exp\!\left(
    \frac{1}{2t^2(1+t^2)}\,
    \bigl\|U_k^\top \boldsymbol{x}_t\bigr\|_2^2
  \right)
}{
  \displaystyle
  \sum_{i=1}^{K}
  \exp\!\left(
    \frac{1}{2t^2(1+t^2)}\,
    \bigl\|U_i^\top \boldsymbol{x}_t\bigr\|_2^2
  \right)
}
\,U_k U_k^\top \boldsymbol{x}_t,$$

$$\bar{\boldsymbol{x}}_\theta(t, \boldsymbol{x}_t, y) = \frac{1}{1+t^2} U_y U_y^\top \boldsymbol{x}_t,$$

$$\bar{\boldsymbol{x}}^{\text{CFG, ideal}}(t, \boldsymbol{x}_t, y) = \frac{1}{1+t^2}\left( (1-\gamma)\sum_{k=1}^{K} \frac{\exp\!\left(\frac{1}{2t^2(1+t^2)}\|\boldsymbol{U}_k^\top \boldsymbol{x}_t\|_2^2\right)}{\sum_{i=1}^{K} \exp\!\left(\frac{1}{2t^2(1+t^2)}\|\boldsymbol{U}_i^\top \boldsymbol{x}_t\|_2^2\right)}\,\boldsymbol{U}_k\boldsymbol{U}_k^\top + \gamma\,\boldsymbol{U}_y\boldsymbol{U}_y^\top \right)\boldsymbol{x}_t$$

We can represent both $\,\bar{\boldsymbol{x}}_\theta(t, \boldsymbol{x}_t, \varnothing)$ and $\bar{\boldsymbol{x}}_\theta(t, \boldsymbol{x}_t, y) $ by the following operator:

$$
(x_t, \boldsymbol{v}) \mapsto
\sum_{k=1}^{K}
\frac{
  \exp\!\left(
    \dfrac{1}{2t^2(1+t^2)}\,
    \bigl|x_t^\top U_k U_k^\top \boldsymbol{v}\bigr|
  \right)
}{
  \displaystyle\sum_{i=1}^{K}
  \exp\!\left(
    \dfrac{1}{2t^2(1+t^2)}\,
    \bigl|x_t^\top U_i U_i^\top \boldsymbol{v}\bigr|
  \right)
}
\,U_k U_k^\top x_t
$$

This operator has a strong resemblance of cross-attention layer. The role of sigal $y$ can be considered as guidance that encourage the denoising process of $x_t$ towards more the selected basis $U_y$. It steers the iterative denoising process towards the conditioning class and away from the previous iterate if $x_t$ initially is not very correlated to the selected basis. 

In short, conditional inference is made possible if the following are fulfilled:
1. Embeddings of $y$ are correlated with $U_y$ that match semantically the desired $x$ you want to obtain

# Class conditioned sampling vs conditional sampling with measurement matching
Recalled that learned conditional posterior denoisers with measurement matching can be written as

$$
\bar{x}_{\theta}(t,\xi,\nu)=\bar{x}_{\theta}(t,\xi)+t^{2}\nabla_{\xi}\log p_{y\mid x}\!\left(\nu\mid\bar{x}_{\theta}(t,\xi)\right).
$$

Let's focus on the comparison of the measurement matching term with the conditional denoiser with classifier guidance.

Measurement matching term: $t^{2}\nabla_{\xi}\log p_{y\mid x}\!\left(\nu\mid\bar{x}_{\theta}(t,\xi)\right)$



# Is the above enough for causal reasoning?

# Questions
The example used above is class-conditioned sampling. What about the conditional sampling for measurement matching?