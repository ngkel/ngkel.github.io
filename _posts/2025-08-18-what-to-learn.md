---
layout: distill
title: What to learn and introduction to problmes solvable by analytical approach
description: Idealistic models that inspire deep network structures
tags: distill formatting
giscus_comments: true
date: 2025-08-22
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

# What learning is all about? Or at least, what are modern methods actually doing?

In nature, data in high dimensional space has low dimensional support. Data lie precisely on geomoetric structures such as subspaces or surfaces. Such constraint make data highly dependent on one another. In other words, natural data is predictable. Completion is perhaps another name of prediction. Denoise and error correction, are tasks that are enabled by the fact that data has constraints.

Thus, we can say that learning is useful as it aims at, first, identifying and representing the low-dimensional structure of data. Next, we utilize the low-dimensional structure to inference for useful downstream tasks.

Our questions then become:
1. How to learn and represent the data
2. How to make use of the representation obtained from 1 to conduct useful inference.

***In this series of blogs, we tackle these questions by claiming that we learn and represent the data as distribution $p(x)$ with low dimenstional structure. Then we make use of $p(x)$ for inferencing.***

# Learning problems that are solvable by analytical approach
Analytical approaches involve explicit low-dimensional structure being assumed while empirical do not. In this section, we discuss classical problem settings and solutions for learning models with assumption that data having geometrically (nearly, piece-wise) linear structures and statistically independent components. Such model is what we call analytical models. Such assumptions offer us efficient algorithm with provable efficiency guarantees for processing data at scale. You may ask why we are learning these models when the real world data is far more complex? We will see in the upcoming blogs that many modern deep learning architectures for complex data actually have structures that are similar to algorithm for modeling data with linear and independent structures, such as overcomplete dictionary learning.

Learning distribution with idealistic, linear models with independent structures has efficient solutions. Examples include:

1. PCA
2. ICA
3. Sparse Coding and Overcomplete Dictionary Learning

Distributions for the above modeling problems can be learned and represented explicitly. We will see later that general distribution does not yield efficient analytical solution and cannot be represented explicitly but they can be learnt implicitly as a denoiser.

***Reminder on what we mean by explicit family of parametric model: The model class that is directly and mathematically specified, for example, "all Gaussians with arbitary mean and covariance". When learning this type of model, we fix the form of the model and tune the set of parameters. Neural networks on the other hands, are considered empirical approaches without clearly stating analytically the form of distribution they are trying to learn. One might argue that the network architecture may limit or define the type of distribution they are able to learn, but the design process of neural networks does not involve explicitly define the form of distribution with clear mathematical descriptions.***

## PCA - finding the single subspace that best fits the data

In PCA, data is assumed, explicitly, to live on a single Gaussian subspace $\mathcal{S} \subseteq \mathbb{R}^D$  of dimension $d$ with basis represented by an orthonormal matrix $\mathbf{U} \in O(D, d) \subseteq \mathbb{R}^{D \times d}$ such that the columns of $\mathbf{U}$ span $\mathcal{S}$. The problem of learning thus becomes:

Given observed data $\{x_i\}_{i=1}^N$, finding the orthonoraml matrix $\mathbf{U} \in O(D,d)$ such that

$$\begin{align}
\mathbf{x}_i = \mathbf{U} \mathbf{z}_i + \boldsymbol{\varepsilon}_i, \quad \forall i \in [N]
\end{align}$$

where

$$\begin{align}
\{\mathbf{z}_i\}_{i=1}^N \subseteq \mathbb{R}^d, \quad d \ll D
\end{align}$$

The optimization problem for finding the optimal subspace $\mathbf{U}^*$ can be formulated as:

$$\begin{align}
\arg \min_{\tilde{\mathbf{U}}} \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{x}_i - \tilde{\mathbf{U}} \tilde{\mathbf{U}}^T \mathbf{x}_i \|_2^2 &= \arg \max_{\tilde{\mathbf{U}}} \frac{1}{N} \sum_{i=1}^{N} \| \tilde{\mathbf{U}}^T \mathbf{x}_i \|_F^2 \\
&= \arg \max_{\tilde{\mathbf{U}}} \text{tr} \left\{ \tilde{\mathbf{U}}^T \left( \frac{\mathbf{X} \mathbf{X}^T}{N} \right) \tilde{\mathbf{U}} \right\}
\end{align}$$

That is, the optimal solution $(\mathbf{U}^*, \{\mathbf{z}^*_i\}_{i=1}^N)$ to the above optimization problem has $\mathbf{z}^*_i = (\mathbf{U}^*)^T \mathbf{x}_i$. That's why the optimization problem above becomes a problem over $\mathbf{U}$ only. The solution of the above optimization problem is given by the top $d$ eigenvectors of:

$$\begin{align}
\frac{\mathbf{X} \mathbf{X}^T}{N} 
\end{align}$$

It is worth noting that projection matrix:

$$\begin{align}
\mathbf{U}^* (\mathbf{U}^*)^T  \approx \mathbf{U} \mathbf{U}^T 
\end{align}$$

can project the noisy data point $\mathbf{x}_i$ onto subspace $\mathcal{S}$.

Reminder of linear algebra concept: Matrix multiplication, let's say multiplied by $\mathbf{U}$ has 3 interpretations:

1. Transformation of a vector
2. Express original vector $\mathbf{x}$ in terms of a vector based on coordinate provided by $\mathbf{U}$.

## Power Iteration - Workhorse of PCA

Computing full SVD for a matrix is computationally intensive. There is, however, an efficient way to comput the eigenvector of a symmetric, positive semidefinite matrix $\mathbf{M}$. The method is called power iteration. 

Power iteration is a fundamental algorithm for finding the dominant eigenvector of a matrix $\mathbf{M}$. The key insight is that repeatedly applying the matrix and normalizing will converge to the eigenvector corresponding to the largest eigenvalue.

**The Algorithm.** 

Assume that $\lambda_1 > \lambda_i$ for all $i > 1$. We want to find the fixed point of:

$$\begin{align}
\mathbf{w} = \frac{\mathbf{M}\mathbf{w}}{\|\mathbf{M}\mathbf{w}\|_2}
\end{align}$$

We can solve this using the following iterative procedure:

$$\begin{align}
\mathbf{v}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad \mathbf{v}_{t+1} = \frac{\mathbf{M}\mathbf{v}_t}{\|\mathbf{M}\mathbf{v}_t\|_2}
\end{align}$$

**Why does this work?**

**Proof Sketch.**  

For any iteration $t$, we have:

$$\begin{align}
\mathbf{v}_t = \frac{\mathbf{M}^t \mathbf{v}_0}{\|\mathbf{M}^t \mathbf{v}_0\|_2}
\end{align}$$

Let's decompose the initial vector $\mathbf{v}_0$ in the eigenbasis of $\mathbf{M}$:

$$\begin{align}
\mathbf{v}_0 = \sum_{i=1}^D \alpha_i \mathbf{w}_i
\end{align}$$

where $\mathbf{M}\mathbf{w}_i = \lambda_i \mathbf{w}_i$ and $\lambda_1 > \lambda_2 \geq \dotsb \geq \lambda_D \geq 0$.

Substituting this into our iteration formula:

$$\begin{align}
\mathbf{v}_t = \frac{ \sum_{i=1}^D \lambda_i^t \alpha_i \mathbf{w}_i }{ \left\| \sum_{i=1}^D \lambda_i^t \alpha_i \mathbf{w}_i \right\|_2 }
\end{align}$$

As $t \to \infty$, since $\lambda_1 > \lambda_i$ for $i > 1$, the terms with $i > 1$ vanish exponentially faster than the first term. This gives us:

$$\begin{align}
\lim_{t \to \infty} \mathbf{v}_t = \frac{ \alpha_1 \mathbf{w}_1 }{ |\alpha_1| } = \operatorname{sign}(\alpha_1)\mathbf{w}_1
\end{align}$$

Therefore, $\mathbf{v}_t$ converges to a unit eigenvector of $\mathbf{M}$ corresponding to the largest eigenvalue.

**Estimating the eigenvalue.**

Once we have the eigenvector, the corresponding eigenvalue can be estimated as:

$$\begin{align}
\lambda_1 \approx \mathbf{v}_t^\top \mathbf{M} \mathbf{v}_t
\end{align}$$

This quantity converges to $\lambda_1$ at the same rate as the eigenvector convergence.

Implementation:

{% highlight c++ linenos %}

import numpy as np

def power_iteration(M, num_iterations=1000, tol=1e-6):
    """
    Compute the dominant eigenvalue and eigenvector of a symmetric positive semidefinite matrix
    using the Power Iteration method.

    Parameters:
    M (numpy.ndarray): Symmetric positive semidefinite matrix
    num_iterations (int): Maximum number of iterations
    tol (float): Convergence tolerance

    Returns:
    tuple: (eigenvalue, eigenvector)
    """
    # Input validation
    if not np.allclose(M, M.T):
        raise ValueError("Matrix must be symmetric")
    if not np.all(np.linalg.eigvals(M) >= -tol):  # Allow small negative values due to numerical errors
        raise ValueError("Matrix must be positive semidefinite")

    # Initialize random vector
    n = M.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)  # Normalize initial vector

    # Power iteration
    for _ in range(num_iterations):
        # Matrix-vector multiplication
        v_new = M @ v

        # Compute eigenvalue (Rayleigh quotient)
        eigenvalue = np.dot(v, v_new)

        # Normalize the new vector
        v_new_norm = np.linalg.norm(v_new)
        if v_new_norm < tol:  # Check for zero vector
            raise ValueError("Power iteration converged to zero vector")
        v_new = v_new / v_new_norm

        # Check convergence
        if np.linalg.norm(v - v_new) < tol or np.linalg.norm(v + v_new) < tol:
            break

        v = v_new

    return eigenvalue, v

# Example usage

# Create a sample 3x3 symmetric positive semidefinite matrix
M = np.array([[4, 2, 0],
              [2, 5, 1],
              [0, 1, 3]])

# Run power iteration
eigenvalue, eigenvector = power_iteration(M)

print("Dominant eigenvalue:", eigenvalue)
print("Corresponding eigenvector:", eigenvector)

# Verify result
print("\nVerification:")
print("M @ eigenvector:", M @ eigenvector)
print("eigenvalue * eigenvector:", eigenvalue * eigenvector)

{% endhighlight %}


## Limitation of PCA

PCA can denoise data by projecting original data on a single principal subspace. However, it has 2 key limitations:

1. It fails to model nonlinear structure
2. Real world data comes from multiple subspaces or surfaces. PCA also fails to model.

The following 3D interactive plots demonstrate visually and computationally why PCA is limited when the data lies on a nonlinear manifold (e.g., a sine wave in 3D). The PCA algorithm is applied to the noisy data, attempting to find the best linear subspace that fits the data. PCA projects the data onto the top 2 principal components, which are linear. The denoised result is obtained by reconstructing the data from this 2D linear subspace. However, the learned model does not offer generative model for nonlinear distribution, since only variance along the principal direction is captured. That is not enough to describe a sinusoidal relationship.

<div class="l-page" style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 400px;">
    <iframe src="{{ '/assets/plotly/blog_2025_08_18/pca_sin_wave_noisy.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
  </div>
  <div style="flex: 1; min-width: 400px;">
    <iframe src="{{ '/assets/plotly/blog_2025_08_18/pca_sin_wave_denoised.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
  </div>
</div>

<div>
    <iframe src="{{ '/assets/plotly/blog_2025_08_18/pca_sin_wave_generative_model.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>

## Mixtures of Subspaces and Sparsely-Used Dictionaries
In mixture of Gaussians, the random variable is generated by, randomly select a Gaussian from K Gaussians, then randomly select a random variable in that selected Gaussian.

The probability density function can be written as:
$$\begin{align}
p(\mathbf{x}_n) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
\end{align}$$

Please don't confuse this with superposition:
$$\begin{align}
\mathbf{x} = \sum_{i=1}^{n} w_i \mathbf{x}_i, \quad \mathbf{x}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{U}_i \mathbf{U}_i^\top)
\end{align}$$

After assuming a mixture of Gaussian model, our task is to learn $\mathbf{U}$ which satisfy the following:

$$\begin{align}
\mathbf{x} = \begin{bmatrix}
| & & | \\
\mathbf{U}_1 & \cdots & \mathbf{U}_K \\
| & & |
\end{bmatrix}
\begin{bmatrix}
\mathbf{z}_1 \\
\vdots \\
\mathbf{z}_K
\end{bmatrix}
= \mathbf{U}\, \mathbf{z}, \quad
\left\|
\begin{bmatrix}
\|\mathbf{z}_1\|_2 \\
\vdots \\
\|\mathbf{z}_K\|_2
\end{bmatrix}
\right\|_0 = 1
\end{align}$$

- $$\mathbf{U} \in \mathbb{R}^{D \times (K d)}$$ is a *dictionary* consisting of the collection of codewords $$\{\mathbf{U}_i\}_{i=1}^K$$.
- $$\mathbf{z} \in \mathbb{R}^{K d}$$ is a *lifted vector* that is $$d$$-sparse, with support on one block of size $$d$$.


where $\mathbf{U} \in \mathbb{R}^{D \times Kd}$

A relaxation is to assume matrix $\mathbf{U} \in \mathbb{R}^{D \times m}$ where $m$ may be smaller or larger than $D$. This leads to sparse dictionary learning problem. There are both geometric and physical/modeling motivations for considering $d \ll m$. The problem is thus turned to be an overcomplete dictionary learning problem. The motivation includes having a richer mixtures of subspaces and some computational experiments in the past reveals the additional modeling power conferred by an overcomplete representation, for example, Bruno Olshausen's paper on overcomplete dictionary learning.

## Overcomplete dictionary learning
To learn $\mathbf{U}$, the corresponding optimization problem can be written as follow:

$$\begin{align}
\min_{\tilde{\mathbf{Z}},\, \tilde{\mathbf{A}}: \|\tilde{\mathbf{A}}_j\|_2 \leq 1} \Big\{\, \|\mathbf{X} - \tilde{\mathbf{A}}\, \tilde{\mathbf{Z}}\,\|_F^2 + \lambda \|\tilde{\mathbf{Z}}\|_1 \, \Big\}
\end{align}$$

There are multiple things worth a separate blog to discuss in this optimization problem:

1. The constraint of having norm in each basis vector of matrix $\mathbf{A}$
2. Such minimization problem is non-convex. How to minimize it with alternating minization of $\mathbf{Z}$ and $\mathbf{A}$?

Despite the dictionary learning problem being a nonconvex problem, it is easy to see that fixing one of the 2 unknowns and optimize the other makes the problem convex and easy to solve. It has been shown that alternating minimization type algorithms indeed converge to the correct solution, at least locally. The algorithm is as follow:

$$\begin{align}
\mathbf{Z}^{\ell+1} = S_{\eta\lambda}\left(\mathbf{Z}^{\ell} - 2 \eta\, \mathbf{A}^{+\top} \left(\mathbf{A}^+ \mathbf{Z}^{\ell} - \mathbf{X}\right)\right),\quad \mathbf{Z}^1=\mathbf{0},\quad \forall\, \ell \in [L]
\end{align}$$

$$\begin{align}
\mathbf{Z}^+ = \mathbf{Z}^L
\end{align}$$

$$\begin{align}
\mathbf{A}^{t+1} = \mathrm{proj}_{\|(\cdot)_j\|_2 \leq 1,\, \forall j}\left(\mathbf{A}^t - 2\nu\, (\mathbf{A}^t \mathbf{Z}^+ - \mathbf{X}) (\mathbf{Z}^+)^\top \right)
\end{align}$$

$$\begin{align}
(\mathbf{A}^1)_j \sim \mathrm{i.i.d.}\, \mathcal{N}\left( \mathbf{0}, \frac{1}{D}\mathbf{I} \right),\quad \forall j \in [m] ,\quad \forall t \in [T]
\end{align}$$

$$\begin{align}
\mathbf{A}^+ = \mathbf{A}^T
\end{align}$$

The main insight from the above algorithm is that, when we fix $\mathbf{A}$, the ISTA update of $\mathbf{Z}^{\ell}$ looks like the forward pass of neural networks. Then the using the thrid line to update $\mathbf{A}$ based on the residual of using the current estimate of the sparse codes $\mathbf{Z}$

# Learned LISTA
The above deep-network interpretation of the alternating miniziation is more conceptual than practical, as the process can be rather inefficient and take many layers or iterations to converge. This is because we are trying to infer both $\mathbf{A}$ and $\mathbf{Z}$ from $\mathbf{X}$. The problem can be simplified in a supervised setting where $(\mathbf{X}, \mathbf{Z})$ are provided and use, say, back propagation type of algorithm to learn $\mathbf{A}$.

The same methodology can be used as a basis to understand the representations computed in other network architectures, such as Transformer. Modern unsupervised learning paradigms are generally more data friendly but still, LISTA algorithm provide a useful practical basis for us to interpret the features in pretrained large-scale deep networks.

The connection between low-dimensional-structure-seeking optimization algorithms and deep network architecture suggests scalable and natural neural learning architectures which may even be usable without backpropagation.