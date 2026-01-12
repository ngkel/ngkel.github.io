---
layout: distill
title: Compression via denoising
description: Idealistic models that inspire deep network structures
tags: representation_learning
giscus_comments: true
date: 2025-09-03
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

# Complexity of a data distribution

The goal of learning is to find the "simplest way" to generate a given set of data. In computer science, how simple the generation methods are is given by the expected number of binary bits required to represent the dataset using the encoding scheme or distribution. Mathematically speaking, given N data points from, let's say, discrete distribution $p(\mathbf{x_i})$:

$$\{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$$
where $$\mathbf{x}_i \sim p(\mathbf{x}_i)$$

The complexity of the target distribution $p(\mathbf{x_i})$ is measured by entropy:

$$\begin{align}
H(\mathbf{x}) \coloneqq \mathbb{E}[\log 1/p(\mathbf{x})] = - \sum_{i=1}^{N} p(\mathbf{x}_i) \log p(\mathbf{x}_i)
\end{align}$$

Please be aware that we have not talked about learnt distribution, which would be represented by the notation $q(\mathbf{x}_i)$ in the future. All we measure by the entropy formula provided above is the complexity of the target distribution which can be "complicated" or "simple" but we are still aiming at learning them. But one thing we can be quite sure is that the distributions of interest, especially of real world data, are nearly low-dimensional. Therefore we can expect that their entropy should be small.

# Minizing coding rate（i.e. entropy）
Remeber that we want to recover the target distribution $p(\mathbf{x})$ from a set of sample $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_N] \in \mathbb{R}^{D \times N}$. There are 2 approaches:

1. Starting with a general distribution (say a normal distribution) and gradually transforming the distribution towards the distribution of data by reducing entropy.

2. Search among a large family of distributions with explicit coding schemes that encode the data with lower coding rate.

## Diffusion and Denoising Process
One way to learn a data distribution is to find a denoiser $\bar{x}^*(t, \cdot)$. The learnt denoiser can be deemed as a proxy to the distribution if we can use it to, starting from a template distribution with no influence from the target distribution, denoise iteratively towards the distribution of $\mathbf{x}$. What does the noisy distribution look like? It can be in a form:

$$\begin{align}
\mathbf{x}_t \dot{=} \mathbf{x} + t\mathbf{g}, \quad \forall t \in [0, T]
\end{align}$$

Once we obtained the denoiser, we can start from a high entropy distribution and sample from it, then gradually transform it to a data point that is close to the target distribution. The $\bar{x}^*(t, \cdot)$ has another form or math meaning, which is (not limited to (2)): 

$$\begin{align}
\bar{\mathbf{x}}^*(t, \boldsymbol{\xi}) \coloneqq \mathbb{E}[\mathbf{x} \mid \mathbf{x}_t = \boldsymbol{\xi}]
\end{align}$$

If the noising process follow the (2), then 
$$\begin{align}
\mathbb{E}[\mathbf{x} \mid \mathbf{x}_t] = \mathbf{x}_t + t^2 \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)
\end{align}$$

If we change (2) to become a more general one,
$$\begin{align}
\dot{\mathbf{x}}_t \triangleq \alpha_t \mathbf{x} + \sigma_t \mathbf{g}, \quad \forall t \in [0, T]
\end{align}$$
then (4) becomes

$$\begin{align}
\mathbb{E}[\mathbf{x} \mid \mathbf{x}_t] = \frac{1}{\alpha_t} (\mathbf{x}_t + \sigma_t^2 \nabla \log p_t(\mathbf{x}))
\end{align}$$

### The form of denoiser for GMM

Let's assume we are given the target distribution which is a Gaussian Mixture Model (GMM). 

$$\begin{align}
\mathbf{x} \sim \sum_{k=1}^K \pi_k \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k),
\end{align}$$

Yes, we are not talking about learning here because the distribution is already given. The corresponding 1-step denoiser is in fact in a very interesting form:

$$\begin{align}
\bar{\mathbf{x}}^*(t, \mathbf{x}_t) = \sum_{k=1}^K \frac{\pi_k \varphi(\mathbf{x}_t; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k + t^2 \mathbf{I})}{\sum_{i=1}^K \pi_i \varphi(\mathbf{x}_t; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i + t^2 \mathbf{I})} \cdot (\boldsymbol{\mu}_k + \boldsymbol{\Sigma}_k(\boldsymbol{\Sigma}_k + t^2 \mathbf{I})^{-1}(\mathbf{x}_t - \boldsymbol{\mu}_k))
\end{align}$$

To understand the it intuitively, let's assume we only have one component in this GMM, the denoiser becomes:

$$\begin{align}
\bar{\mathbf{x}}^*(t, \mathbf{x}_t) = \boldsymbol{\mu} + \boldsymbol{\Sigma}(\boldsymbol{\Sigma} + t^2 \mathbf{I})^{-1}(\mathbf{x}_t - \boldsymbol{\mu}) = \boldsymbol{\mu} + \mathbf{V} \begin{bmatrix} \lambda_1 / (\lambda_1 + t^2) \\ \ddots \\ \lambda_D / (\lambda_D + t^2) \end{bmatrix} \mathbf{V}^T(\mathbf{x}_t - \boldsymbol{\mu})
\end{align}$$

We can translate this to English as:

1. Translate the input $x_t$ by $\mu$.
2. Contract the (translated) input $x_t-\mu$ in each eigenvector direction by a quantity $\lambda_i / (\lambda_i + t^2)$. If the translated input is low-rank and some eigenvalues of $\boldsymbol{\Sigma}$ are zero, these directions get immediately.
contracted to 0 by the denoiser, ensuring that the output of the contraction is similarly low-rank.
3. Translate the output back by $\mu$.

It might appear very similar to Power Iteration, however, they are fundamentally different. Power iteration implements a contraction mapping towards a subspace—namely the subspace spanned by the first principal component. In contrast, the
iterates in the denoiser converge to the mean $\mu$ of the underlying distribution, which is a single point.

### Yet in reality, we have are not given the target distribution and we need to learn it...

Without prior knowledge, such as it is given as a GMM model with parameters:

$$\begin{align}
\begin{bmatrix} \mathbf{x} \\ \mathbf{x}_t \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \boldsymbol{\mu}_y \\ \boldsymbol{\mu}_y \end{bmatrix}, \begin{bmatrix} \boldsymbol{\Sigma}_y & \boldsymbol{\Sigma}_y \\ \boldsymbol{\Sigma}_y & \boldsymbol{\Sigma}_y + t^2 \mathbf{I} \end{bmatrix} \right)
\end{align}$$

we cannot derive the explicit form of denoiser $\bar{\mathbf{x}}^*(t, \boldsymbol{\xi})$ since we don't know the $p_t$. We need to learn $\bar{\mathbf{x}}^*(t, \boldsymbol{\xi})$ from data. Recall that the denoiser is defined as minimizing the mean-squared error $$\begin{align}
\mathbb{E}[\|\bar{\mathbf{x}}(t, \mathbf{x}_t) - \mathbf{x}\|_2^2]
\end{align}$$

We can learn the denoiser by 

$$\begin{align}
\min_{\theta \in \Theta} \mathbb{E}_{x, x_t}[\|\bar{\mathbf{x}}_\theta(t, \mathbf{x}_t) - \mathbf{x}\|_2^2]
\end{align}$$

In which ${\mathbf{x}}_\theta(t, \mathbf{x}_t)$ is usually a neural network. 

### What should be a good model architecture for ${\mathbf{x}}_\theta(t, \mathbf{x}_t)$?

Let's assume the model to be learnt is a Gaussian Mixture Model GMM and revisit the denoiser form 

$$
\bar{\mathbf{x}}^*(t, \mathbf{x}_t) = \sum_{k=1}^K \frac{\pi_k \varphi(\mathbf{x}_t; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k + t^2 \mathbf{I})}{\sum_{i=1}^K \pi_i \varphi(\mathbf{x}_t; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i + t^2 \mathbf{I})} \cdot (\boldsymbol{\mu}_k + \boldsymbol{\Sigma}_k(\boldsymbol{\Sigma}_k + t^2 \mathbf{I})^{-1}(\mathbf{x}_t - \boldsymbol{\mu}_k))
$$

This functional form is similar to attention mechanism in Transformer! If we assume the mean being zero, we can rewrite it as 

$$\begin{align}
\bar{\mathbf{x}}^*(t, \mathbf{x}_t) = \sum_{k=1}^K \frac{\varphi(\mathbf{x}_t; \mathbf{0}, \mathbf{U}_k \mathbf{U}_k^\top + t^2 \mathbf{I})}{\sum_{i=1}^K \varphi(\mathbf{x}_t; \mathbf{0}, \mathbf{U}_i \mathbf{U}_i^\top + t^2 \mathbf{I})} \cdot (\mathbf{U}_k \mathbf{U}_k^\top (\mathbf{U}_k \mathbf{U}_k^\top + t^2 \mathbf{I})^{-1} \mathbf{x}_t)
\end{align}$$

If we further simplify it as $K = 1$ the learning task will thus become parameterizing the denoiser   

$$\begin{align}
\bar{\mathbf{x}}(t, \mathbf{x}_t) = \frac{1}{1+t^2} \mathbf{V}\mathbf{V}^\top \mathbf{x}_t
\end{align}$$

where $\mathbf{V} \in O(D, P)$ are learnable parameters.  

Substituting this to the training loss (12) would equivalent to solving 


$$\begin{align}
\min_{V \in O(D, P)} \mathbb{E}_x [\|\mathbf{x} - \frac{1}{1+t^2} \mathbf{V}\mathbf{V}^\top \mathbf{x}\|_2^2] = \mathbb{E}_x [\|\mathbf{x}\|_2^2] + \left(\left(\frac{1}{1+t^2}\right)^2 - \frac{2}{1+t^2}\right) \mathbb{E}_x [\|\mathbf{V}^\top \mathbf{x}\|_2^2]
\end{align}$$

and futher equivalent to 

$$\begin{align}
\max_{V \in O(D, P)} \mathbb{E}_x [\|\mathbf{V}^\top \mathbf{x}\|_2^2]
\end{align}$$

This is exactly the probablistic PCA problem.

### Lesson from optimization tells us that 1 big step is not enough

Similar to how one step of gradient descent is almost never suﬀicient to minimize an objective in practice
when initializing far from the optimum, the output of the Bayes-optimal denoiser $\bar{\mathbf{x}}^*(t, \cdot)$ is almost never contained in a high-probability region of the data distribution when $t$ is large, especially when the data have low-dimensional structures. Analgously to gradient descent with decaying step size, we use denoiser multiple times iteratively as follow:

$$\begin{align}
\hat{\mathbf{x}}_{t_{\ell-1}}  = \left(1 - \frac{1}{\ell}\right) \cdot \hat{\mathbf{x}}_{t_{\ell}} + \frac{1}{\ell} \cdot \bar{\mathbf{x}}^*(t_{\ell}, \hat{\mathbf{x}}_{t_{\ell}}).
\end{align}$$

### Recap of key ideas
Let's summarize the basic scheme of diffusion and denoising.
The basic diffusion process is given by:

#### Diffusion
$$
\mathbf{x}_t \dot{=} \mathbf{x} + t\mathbf{g}, \quad \forall t \in [0, T]
$$

Since, $$\begin{align}
\frac{\mathbf{x}_T}{T} = \frac{\mathbf{x} + T\mathbf{g}}{T} = \frac{\mathbf{x}}{T} + \mathbf{g} \rightarrow \mathbf{g} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
\end{align}$$

Then,
$$\begin{align}
\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, T^2 \mathbf{I})
\end{align}$$

#### Sampling and denoiser

Sample $\hat{\mathbf{x}}_T \sim \mathcal{N}(\mathbf{0}, T^2 \mathbf{I})$ (i.i.d. of everything else)

Discretizing $[0, T]$ into $0 = t_0 < t_1 < \ldots < t_L = T$ uniformly using $t_\ell = T\ell/L$.

Run denoising iteration
$$
\hat{\mathbf{x}}_{t_{\ell-1}} = \left(1 - \frac{1}{\ell}\right) \cdot \hat{\mathbf{x}}_{t_\ell} + \frac{1}{\ell} \cdot \bar{\mathbf{x}}^*(t_\ell, \hat{\mathbf{x}}_{t_\ell})
$$


### Different ways to diffues and denoise
Different noise model. (19) is a worse approximation in high-dim, anisotropic space. In other words, the distance increased between $\mathcal{N}(\mathbf{0}, T^2 \mathbf{I})$ and the true $\mathbf{x}_{t_\ell}$ therefore we come up with a different noise model called variance prerserving process:

$$
\dot{\mathbf{x}}_t \triangleq \alpha_t \mathbf{x} + \sigma_t \mathbf{g}, \quad \forall t \in [0, T]
$$ 

where

$$\begin{align}
T = 1 \\
\alpha_t &= \sqrt{1 - t^2} \text{ and } \sigma_t = t \\
\mathbf{x}_1 &\sim \mathcal{N}(\mathbf{0}, \mathbf{I}) 
\end{align}$$

and the denoising iteration becomes

$$\begin{align}
\hat{\mathbf{x}}_{t_{\ell-1}} = \frac{\sigma_{t_{\ell-1}}}{\sigma_{t_{\ell}}} \hat{\mathbf{x}}_{t_{\ell}} + \left(\alpha_{t_{\ell-1}} - \frac{\sigma_{t_{\ell-1}}}{\sigma_{t_{\ell}}}\alpha_{t_{\ell}}\right) \bar{\mathbf{x}}^*(t_{\ell}, \hat{\mathbf{x}}_{t_{\ell}})
\end{align}$$

Also, we train one neural network for all time steps instead of each timestep instead of one for each timestep.

$$\begin{align}
\min_{\theta} \mathbb{E}_{t, x, x_t} [\|\bar{\mathbf{x}}_\theta(t, \mathbf{x}_t) - \mathbf{x}\|_2^2]
\end{align}$$


## Inferecne with learnt model

