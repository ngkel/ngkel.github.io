---
layout: distill
title: Compression via denoising
description: Idealistic models that inspire deep network structures
tags: representation_learning
giscus_comments: true
date: 2026-03-03
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
  - name: Complexity of a data distribution
  - name: Minizing coding rate（i.e. entropy）
    subsections:
      - name: Diffusion and Denoising Process
        subsections:
          - name: The form of denoiser for GMM
          - name: Yet in reality, we have are not given the target distribution and we need to learn it...
          - name: What should be a good model architecture for the denoiser to be learnt?
          - name: Lesson from optimization tells us that 1 big step is not enough
          - name: Recap of key ideas
            subsections:
              - name: Diffusion
              - name: Sampling and denoiser
          - name: Different ways to diffues and denoise
  - name: Inferecne with learnt model

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

The Temporal Straightening for Latent Planning paper brings control theory back to my radar.

# What control theory concerns

1. Stability
2. Controllability
3. Nonlinear dynamic linearization

# Discussion on System Stability

## 1. Setting up the linear system picture

Consider a linear time-invariant (LTI) dynamical system in state space form:

- Continuous time:  
  $$\dot{x}(t) = A x(t)$$
- Discrete time:  
  $$x_{k+1} = A x_k$$

Here $x$ is the state vector and $A$ is an $n \times n$ matrix (the “state” or “transition” matrix). The stability of the origin $x=0$ is determined entirely by the spectral properties (eigenvalues) of $A$.

Before talking stability, it helps to have a very geometric understanding of eigenvalues and eigenvectors of $A$, since they correspond to the system’s “modes”.

To start, what kind of systems are you most interested in for this tutorial: continuous-time $\dot{x} = Ax$, discrete-time $x_{k+1} = Ax_k$, or both?

## 2. Eigenvalues and eigenvectors of $A$: definition and intuition

For a square matrix $A \in \mathbb{R}^{n \times n}$, a (right) eigenpair $(\lambda, v)$ satisfies

$$A v = \lambda v$$

with $v \neq 0$.

- $v$ is an eigenvector: a direction in state space.
- $\lambda$ is an eigenvalue: the factor by which $A$ scales that direction (possibly complex).

Key intuition:

- In general, $A$ maps a vector to some rotated and scaled version.  
- Eigenvectors are the “special” directions that are not rotated, only scaled (and maybe flipped) by $A$.

The collection of eigenvectors (when they form a basis) gives a coordinate system where the dynamics decouple into scalar 1D modes. From a control viewpoint, these eigen-directions are the system’s natural modes; each eigenvalue tells you how its corresponding mode grows, decays, or oscillates.


## 3. Eigen-decomposition and change of coordinates

If $A$ has $n$ linearly independent eigenvectors, you can form the matrix

$$V = [v_1 \ v_2 \ \dots \ v_n]$$

and a diagonal matrix

$$\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$$

such that

$$A = V \Lambda V^{-1}$$

This is the eigendecomposition of $A$.

Interpretation:

- $V$ changes coordinates from the standard basis to the eigenvector basis.
- In the new coordinates $z = V^{-1}x$, the dynamics (for $\dot{x} = Ax$) become

  $$\dot{z} = \Lambda z$$

  i.e., $n$ decoupled scalar linear ODEs $\dot{z}_i = \lambda_i z_i$.

This decoupling is the core reason eigenvalues/eigenvectors are so central: stability reduces to studying scalar exponentials or scalar powers.

## 4. Modal decomposition of trajectories

Definition: mode = eigenvector direction + its time evolution

Using the eigenbasis, solutions to the unforced system can be written as a sum of modes.

### Continuous time

For $\dot{x} = A x$, the solution is

$$x(t) = e^{At} x(0)$$

If $A$ is diagonalizable, then

$$x(t) = \sum_{i=1}^n \alpha_i e^{\lambda_i t} v_i$$

where the coefficients $\alpha_i$ come from expressing $x(0)$ in the eigenvector basis.

### Discrete time

For $x_{k+1} = A x_k$, the solution is

$$x_k = A^k x_0$$

and similarly, if $A$ is diagonalizable,

$$x_k = \sum_{i=1}^n \alpha_i \lambda_i^k v_i$$

Each term is an eigenmode:

- The direction is fixed (the eigenvector $v_i$).
- The amplitude evolves as a scalar exponential $e^{\lambda_i t}$ (continuous) or as a power $\lambda_i^k$ (discrete).

Stability conditions i.e. to be stable for the dynamical system follow directly from the behavior of these scalar factors:

- Continuous time: require all $e^{\lambda_i t} \to 0$, which means all eigenvalues must have strictly negative real part.
- Discrete time: require all $\lambda_i^k \to 0$, which means all eigenvalues must have magnitude strictly less than 1 (Schur stability).

## 5. Example problem in control system

We might be given an unstable system by $A$. The goal might be to add input $Bu$ to drive the system to stable by driving the system to stability by forcing the eigenvalues to negative. 

***

# Discussion on System Linearization

***

# What temporal straightening offers

## Training objectives

There are 2 training objectives in the paper:

1. Prediction objective - 
$$\begin{align}
\mathcal{L}_{\text{pred}} = \big\| \hat{z}_{t+1} - \operatorname{sg}(z_{t+1}) \big\|_{2}^{2} 
\end{align}$$

where $sg(\cdot)$ is stop gradient to prevent representation collapse.

2. Straightening objective -
$$\begin{align}
\mathcal{L}_{\text{curv}} = 1 - C
\end{align}$$

The overall objective is thus:

$$\begin{align}
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} + \lambda \mathcal{L}_{\text{curv}}
\end{align}$$

## The training objectives tries to promote linear dynamics
The above objective and neural network training aim at obtaining linear latent dynamics:

$$\begin{align}
f:(z_t,a_t)\mapsto A z_t + B a_t,\quad \text{s.t.}\quad z_{t+1}=A z_t + B a_t,\quad A\in \mathbb{R}^{d\times d},\ B\in \mathbb{R}^{d\times d_a}
\end{align}$$

On top of this linear dynamics definition, we can characterize a linear dyanmics as $\epsilon$-straight transition if:

$$\begin{align}
\|A-I\|_2\le \epsilon
\end{align}$$

If $\epsilon$ approach 0, the state evolves linearly along a straight-line trajectory modified only by the control input. i.e.:

$$\begin{align}
g:(z_t,a_t)\mapsto z_t + B a_t
\end{align}$$

If you hold $a_{t}$ over time constant, then you get a straight-line trajectory in latent space. If the direction $a_{t}$ can change at each step, the overall trajectory is still curved but piece-wise straight.

## Planning problem benefits from linear dynamics

Recall that in control theory we concerns:

1. Controllability   

### Conditioning of Planning Hessian under Linear Dynamic with quadratic loss (not necessarily $\epsilon$ is small, we will talk about that in the next sub-section)

Remember, in optimization, gradient tells you the local direction of the steepest descent while the Hessian encodes the local curvature, i.e. how the gradient changes as you move in different direction. Eigenvalues and eigenvectors of the Hessian tells you how sharp/flat the loss function is along a direction. The curvature also tells you whether a stationary point is a local minimum, maximum or saddle.

Effective condition number of the Hessian is slightly different condition number of Hessian because it is the maximum eigenvalue of Hessian divided by the smallest positive eigenvalue i.e. the negative eigenvalues are ignored. 

$$\kappa_{\mathrm{eff}}(H) := \sigma_{\max}(H) / \sigma_{\min}^{+}(H)$$

Why? 

Conditioning of the planning Hessian determine how easy and stable the planning/optimization is to solve. The smaller the condition number, the more well-conditioned it is. Condition number depends on the planning Hessian. 

Now we will discuss the characteristic of the planning Hessian under linear dynamic with quadratic loss. We also assume that dimension of state and dimension of action is the same and $B$ is invertible. Unrolling (4) yields:

$$\begin{align}
z_K = A^K z_0 + \sum_{t=0}^{K-1} A^{K-1-t} B a_t
\end{align}$$

The planning Hessain in $a$ for linear dynamic with planning objective being quadratic is positive semi-definite, in the form:

$$\begin{align}
H := \nabla_a^2 \mathcal{L}(a) = 2 J_{\Phi}^{\top} J_{\Phi} \succeq 0
\end{align}$$

where 

$$J_{\Phi} = \bigl[ A^{K-1} B \;\; A^{K-2} B \;\; \dots \;\; B \bigr] \in \mathbb{R}^{d \times Kd}$$

and 

$$J_{\Phi}J_{\Phi}^{\top}$$ 

be the finite-horizon controllability Gramian.

Without specifying $\epsilon$ being small, the planning Hessian of the linear dynamic with respect to quadratic loss, assumed that $d = d_{a}$ and $B$ being invertible, will determine the effective condition number. 

$$\kappa_{\mathrm{eff}}(H) := \sigma_{\max}(H) / \sigma_{\min}^{+}(H)$$

The bound of condition number is also determined by $A, B, K$. It also depends on the finite-horizon controllability Gramian.

$$\kappa_{\mathrm{eff}}(H) = \kappa(\mathcal{W}_{K}) \le \kappa(B)^{2} \, \frac{\sum_{k=0}^{K-1} \sigma_{\max}(A)^{2k}}{\sum_{k=0}^{K-1} \sigma_{\min}(A)^{2k}} \le \kappa(B)^{2} \, \kappa(A)^{2(K-1)}$$

If the transition is $\epsilon-straight$ with $\epsilon = \lVert A - I \rVert_{2} < 1$, then 

$$\kappa_{\mathrm{eff}}(H) \le \kappa(B)^{2} \left( \frac{1 + \varepsilon}{1 - \varepsilon} \right)^{2(K-1)}$$

The math above basically says that $\epsilon$-straight transition control the condition number of the planning Gaussian, when $\epsilon$ is small, the Gramian remains better condition, yielding the effective condition number of planning Hessian grows slowly with the horizon.

Since the linear planning objective is quadratic with Hessian being $\succeq$ 0, gradient descent converges lienarly at a rate controlled by the condition number, so the improved bounds on $\kappa_{\mathrm{eff}}(H)$ translate to faster optimization in practice.

#### Recall what condition number mean

### Conditioning of Planning Hessian under $\epsilon$-straight Linear Dynamics with Quadratic Loss


The paper is specifically interested in regime where $\epsilon$ is small.