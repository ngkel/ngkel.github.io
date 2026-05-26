---
layout: distill
title: Temporal Straightening for Latent Planning in Control Theory Language
description: A revision on control theory basics and insight from temporal straightening paper
tags: representation_learning, reinforcement_learning
giscus_comments: true
date: 2026-05-22
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
  - name: Motivation of this blog
  - name: Why LTI system as the targeted system?
  - name: Discussion on System Linearization
  - name: Discussion on System Stability
    subsections:
      - name: 1. Setting up the linear system picture
      - name: 2. Eigenvalues and eigenvectors of $A$ - definition and intuition
      - name: 3. Eigen-decomposition and change of coordinates
      - name: 4. Modal decomposition of trajectories
      - name: 5. Example problem in control system
  - name: Discussion on Controllability
    subsections:
      - name: Intuition of controllability
      - name: Linear System Dynamics
      - name: The Controllability Matrix
  - name: Discussion On Observability
    subsections:
      - name: Kalman Filter
  - name: Temporal Straightening for Latent Planning Paper in Control Theory Language
    subsections:
      - name: Better condition number of Hessian for more efficient optimization
      - name: Better alignment of the Euclidean distance and geodesic distance between two states, in latent space
      - name: Connection to Classical System Identification
      - name: Training objectives
      - name: The training objectives tries to promote linear dynamics
  - name: My suggestion on future research direction

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

# Motivation of this blog
The Temporal Straightening for Latent Planning paper brings control theory back to my radar. Key ideas of the paper include:
1. Extract a linear time-invariant (LTI) dynamical system from observations in pixel space and action space. 

2. In addition to (1), another key idea is to use a curvature regularizer to encourage locally straightened latent trajectories, thus decreasing the curvature of the resulting embeddings. (This is in fact very similar to the concept of manifold flattening)

3. Straightening encourages the latent Euclidean distance to better align with the geodesic distance.

4. Near perfect reconstruction can be attained with a very low features dimensionality (reduce the embedding dimension from 384 ->8).

5. Straightening makes the task loss landscape closer to convex and better conditioned, making gradient-based optimization easier. (Does straightening make it more convex and better conditioned?)

LTI dynamical system is a special case of linear dynamical system.
$$\begin{align}
\dot{x}(t) = A(t) x(t) + B(t) u(t), \quad y(t) = C(t) x(t) + D(t) u(t)
\end{align}$$
where $A(t),B(t),C(t),D(t)$ are constant $A, B, C, D$ in LTI system.
$$\begin{align}
\dot{x}(t) = A x(t) + B u(t), \quad y(t) = C x(t) + D u(t)
\end{align}$$

Despite the paper does not mention the term "linearization", they are pretty much related. I will walk through some basic concept in control theory and how this temporal straightening paper bridge the gap between classical control theory and deep learning. And we will also look at the straightening criteria, which I was somehow confused with the concept linearization, since a linear system does not necessarily has the trajectory straightened.

# Why LTI system as the targeted system?

In general, all control problems, no matter linear or non-linear, focus on the following 3 aspects:

1. Stability
2. Controllability & Reachability
3. Observability
4. System Identification

Linear systems provide a foundational mathematical language that is simple enough to be analyzed deeply. When a system is linearized, it enables us to use powerful and established mathematical tools to analyse the system, that are otherwise unavailable or vastly more complicated for nonlinear systems. 

But I personally think that "easier" analysis is not the main reason. There should be 2 main reasons. 

1. Structured (maybe linearized) representation allows more efficient computation
2. Structured (maybe linearized) representation facilitate memory construction about the dynamical model of the world.

# Discussion on System Linearization

Linearization begins by identifying an equilibrium or "fixed" point where the system's rate of change is zero. By introducing a small perturbation variable around this fixed point and applying a Taylor series expansion, the nonlinear dynamics can be locally approximated. The zeroth-order term vanishes, leaving a simplified linear system defined by the Jacobian matrix evaluated precisely at that fixed point.

However, transformation from non-linear system to LTI system in temporal straightening paper is more than linearization around a fixed point. It is a global linearization instead of a local one. The paper also additionally demands curvature of the trajectories to be straight.

# Discussion on System Stability

## 1. Setting up the linear system picture

Consider a linear time-invariant (LTI) dynamical system in state space form:

- Continuous time:  
  $$\begin{align}
  \dot{x}(t) = A x(t)
  \end{align}$$
- Discrete time:  
  $$\begin{align}
  x_{k+1} = A x_k
  \end{align}$$

Here $x$ is the state vector and $A$ is an $n \times n$ matrix (the “state” or “transition” matrix). 

Stability is defined as the behavior of a linear dynamical system's state over time, specifically whether it naturally returns to an equilibrium point or diverges.

Before talking stability, it helps to have a very geometric understanding of eigenvalues and eigenvectors of $A$, since they correspond to the system’s “modes”.

## 2. Eigenvalues and eigenvectors of $A$ - definition and intuition

For a square matrix $A \in \mathbb{R}^{n \times n}$, a (right) eigenpair $(\lambda, v)$ satisfies

$$\begin{align}
A v = \lambda v
\end{align}$$

with $v \neq 0$.

- $v$ is an eigenvector: a direction in state space.
- $\lambda$ is an eigenvalue: the factor by which $A$ scales that direction (possibly complex).

Key intuition:

- In general, $A$ maps a vector to some rotated and scaled version.  
- Eigenvectors are the “special” directions that are not rotated, only scaled (and maybe flipped) by $A$.

The collection of eigenvectors (when they form a basis) gives a coordinate system where the dynamics decouple into scalar 1D modes. From a control viewpoint, these eigen-directions are the system’s natural modes; each eigenvalue tells you how its corresponding mode grows, decays, or oscillates.


## 3. Eigen-decomposition and change of coordinates

If $A$ has $n$ linearly independent eigenvectors, you can form the matrix

$$\begin{align}
V = [v_1 \ v_2 \ \dots \ v_n]
\end{align}$$

and a diagonal matrix

$$\begin{align}
\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)
\end{align}$$

such that

$$\begin{align}
A = V \Lambda V^{-1}
\end{align}$$

This is the eigendecomposition of $A$.

Interpretation:

- $V$ changes coordinates from the standard basis to the eigenvector basis.
- In the new coordinates $z = V^{-1}x$, the dynamics (for $\dot{x} = Ax$) become

  $$\begin{align}
  \dot{z} = \Lambda z
  \end{align}$$

  i.e., $n$ decoupled scalar linear ODEs $\dot{z}_i = \lambda_i z_i$.

This decoupling is the core reason eigenvalues/eigenvectors are so central: stability reduces to studying scalar exponentials or scalar powers.

## 4. Modal decomposition of trajectories

Definition: mode = eigenvector direction + its time evolution

Using the eigenbasis, solutions to the unforced system can be written as a sum of modes.

### Continuous time

For $\dot{x} = A x$, the solution is

$$\begin{align}
x(t) = e^{At} x(0)
\end{align}$$

If $A$ is diagonalizable, then

$$\begin{align}
x(t) = \sum_{i=1}^n \alpha_i e^{\lambda_i t} v_i
\end{align}$$

where the coefficients $\alpha_i$ come from expressing $x(0)$ in the eigenvector basis.

### Discrete time

For $x_{k+1} = A x_k$, the solution is

$$\begin{align}
x_k = A^k x_0
\end{align}$$

and similarly, if $A$ is diagonalizable,

$$\begin{align}
x_k = \sum_{i=1}^n \alpha_i \lambda_i^k v_i
\end{align}$$

Each term is an eigenmode:

- The direction is fixed (the eigenvector $v_i$).
- The amplitude evolves as a scalar exponential $e^{\lambda_i t}$ (continuous) or as a power $\lambda_i^k$ (discrete).

Stability conditions i.e. to be stable for the dynamical system follow directly from the behavior of these scalar factors:

- Continuous time: require all $e^{\lambda_i t} \to 0$, which means all eigenvalues must have strictly negative real part.
- Discrete time: require all $\lambda_i^k \to 0$, which means all eigenvalues must have magnitude strictly less than 1 (Schur stability).

## 5. Example problem in control system

We might be given an unstable system by $A$. The goal might be to add input $Bu$ to drive the system to stable by driving the system to stability by forcing the eigenvalues to negative. 

***

# Discussion on Controllability

## Intuition of controllability

If the system is controllable, we can put the eigenvalues of the dynamical system anywhere we want to make it stable or unstable. If the system is controllable, we can steer the state $x$ to anywhere we want. This is also called reachability.

We also care about how controllable a system can be by looking at the controllability Gramian.

## Linear System Dynamics

Consider a continuous-time linear time-invariant (LTI) system described by the state-space equations:

$$\begin{align}
\dot{x} = Ax + Bu
\end{align}$$

Where:
- $x \in \mathbb{R}^n$ is the state vector representing the system's internal condition.
- $u \in \mathbb{R}^m$ is the control input vector.
- $A \in \mathbb{R}^{n \times n}$ is the system dynamics matrix.
- $B \in \mathbb{R}^{n \times m}$ is the input control matrix.

## The Controllability Matrix

To determine if a system is controllable, we can use a straightforward algebraic test. We construct the **controllability matrix** $\mathcal{C}$, which is formed by concatenating the column vectors of $B, AB, A^2B, \dots$:

$$\begin{align}
\mathcal{C} = \begin{bmatrix} B & AB & A^2B & \dots & A^{n-1}B \end{bmatrix}
\end{align}$$

For a system with $n$ states, $\mathcal{C}$ will be an $n \times (n \cdot m)$ matrix.

### The Rank Condition

The primary test for controllability is evaluating the rank of this matrix. The system is fully controllable if and only if the controllability matrix $\mathcal{C}$ has full row rank. That is:

$$\begin{align}
\text{rank}(\mathcal{C}) = n
\end{align}$$

If the rank is less than $n$, the system is uncontrollable. The rank of $\mathcal{C}$ tells us the dimension of the controllable subspace. 

### Intuitive Interpretation of the Controllability Matrix

The terms within the controllability matrix represent the directions in the state space that the input can influence:
- $B$: Represents the states that the input $u$ can actuate **directly**.
- $AB$: Represents the states we can actuate **indirectly** after the input propagates through the system dynamics $A$ for a brief moment.
- $A^2B, \dots, A^{n-1}B$: Represent further propagation and coupling of the input through the system's internal dynamics.

By the Cayley-Hamilton theorem, any higher powers of $A$ (i.e., $A^n, A^{n+1}, \dots$) can be expressed as linear combinations of the lower powers up to $A^{n-1}$. Thus, adding more terms beyond $A^{n-1}B$ will not yield any new linearly independent directions in the state space, which is why the controllability matrix terminates there.

### Controllability is Not Binary
In real-world applications, "degrees of controllability" provide a more useful, richer measure, showing how controllable different directions in the state space ($\mathbb{R}^n$) actually are.

### The Controllability Gramian
The controllability Gramian, denoted as $W_T$, is introduced to quantify these degrees. It is defined mathematically as the integral of the matrix exponential and the input matrix:

$$\begin{align}
W_T = \int_0^T e^{A\tau}BB^Te^{A^T\tau} d\tau
\end{align}$$

This yields an $n \times n$ symmetric matrix with positive real eigenvalues. The eigendecomposition of the Gramian reveals that the eigenvectors corresponding to the largest eigenvalues represent the most controllable directions. Practically, this means a unit of control energy can drive the system further along these specific directions than those with smaller eigenvalues.

### Computing with SVD
By taking the Singular Value Decomposition (SVD) of the system's controllability matrix, you can extract the same information. The largest singular values correspond to the most controllable directions, and the left singular vectors form an "energy ellipsoid" defining how far a system can be steered given a unit of input energy. 

### Stabilizability vs. Full Controllability
For very high-dimensional systems (like modeling fluid flow with millions of variables), requiring full $\mathbb{R}^n$ controllability is often unrealistic and unnecessary. Instead, the focus shifts to **stabilizability**. A system is stabilizable if all its unstable or lightly damped eigenvectors are within the controllable subspace. This ensures that any dynamics that could cause the system to fail can be controlled, while highly stable, self-correcting dynamics can be ignored.

### The PBH Test

In control theory, determining whether a system is controllable is a foundational step. While the standard controllability matrix provides one way to check this, the Popov-Belevitch-Hautus (PBH) test offers an extremely simple and insightful alternative . This tutorial explores the PBH test and what it tells us about choosing our input matrix $B$ .

### The PBH Test Condition

For a linear time-invariant system with state matrix $A \in \mathbb{R}^{n \times n}$ and input matrix $B \in \mathbb{R}^{n \times p}$, the pair $(A, B)$ is controllable if and only if the following rank condition holds true :

$$\begin{align}
\text{rank}([A - \lambda I \quad B]) = n
\end{align}$$

for all $\lambda \in \mathbb{C}$ in the complex plane .

### Testing at the Eigenvalues

While checking all complex numbers seems daunting, we can simplify this dramatically . The matrix $(A - \lambda I)$ naturally has full rank $n$ for almost any $\lambda$ . It only drops rank when $\lambda$ is an eigenvalue of $A$ . 

Therefore, we *only* need to test the rank condition at the eigenvalues of $A$ :

$$\begin{align}
\text{rank}([A - \lambda_i I \quad B]) = n \quad \text{for all } \lambda_i \in \text{eig}(A)
\end{align}$$

To understand why the matrix $(A - \lambda I)$ only drops rank when $\lambda$ is an eigenvalue, we must look at the definitions of rank, eigenvectors, and invertible matrices.

Notes: If $\lambda$ is an eigenvalue, the determinent of matrix $(A - \lambda I)$ will be 0 as you may recall that's the way we find the eigenvector.

### Physical Intuition: Components in Eigenvector Directions

When we evaluate the matrix at an eigenvalue $\lambda_i$, the term $(A - \lambda_i I)$ loses rank specifically in the direction of the corresponding eigenvector . 

For the augmented matrix $[A - \lambda_i I \quad B]$ to regain full rank $n$, the input matrix $B$ must complement it . This means $B$ must have some component in every eigenvector direction of matrix $A$ . If an actuator (a column of $B$) is completely orthogonal to an eigenvector, the system cannot control the mode associated with that eigenvector.

### Multiple Actuators and Eigenvalue Multiplicity

The PBH test also reveals the minimum number of actuators (columns of $B$) needed to control a system :

1. **Distinct Eigenvalues**: If all eigenvalues of $A$ are distinct, a single properly placed actuator (a $n \times 1$ vector $B$) is sufficient for controllability, provided it has non-zero components along all eigenvector directions .
2. **Repeated Eigenvalues**: If $A$ has an eigenvalue with a geometric multiplicity $k > 1$ (e.g., a repeated root with multiple independent eigenvectors), the null space of $(A - \lambda I)$ will have dimension $k$ . To fill this null space and achieve rank $n$, $B$ must have at least $k$ independent columns . Thus, a multiplicity of $k$ dictates that you need *at least* $k$ independent control inputs to make the system controllable .

# Discussion On Observability

## Kalman Filter
The most basic way to build a Kalman filter usually requires known dynamical system parameters $A, B, C$ and system noises. Then we have input signal $y, u$. Kalman Filter is basically a denoiser for corrupted data with noise and incomplete measurement given a known model.

***

# Temporal Straightening for Latent Planning Paper in Control Theory Language

If we are to see the Temporal Straightening for Latent Planning paper as a research project to tackle control problems, how should we translate the project in control theory language?

- System identification
- MPC

This paper try to seek latent representation of pixel frames $z_t$ and the dynamic dynamics $f$ such that they can apply gradient-based planning with better planning optimization landscape. They try to seek for more favourable optimization landscape by

1. Having straightened latent dynamic for better proxy of geodesic distance.
2. Better condition number of Hessian

## Better condition number of Hessian for more efficient optimization

The planning optimization problem is written as a finite-horizon goal-reaching control problem with the following MSE loss:

$$\begin{align}
\mathcal{L}(\mathbf{a}) = \left\| z_K - z_g \right\|_2^2, \qquad z_K = \Phi(\mathbf{a}),
\end{align}$$

The latent dynamical constraint is written as:

$$\begin{align}
z_{t+1} = A z_t + B a_t, \qquad A \in \mathbb{R}^{d \times d}, \; B \in \mathbb{R}^{d \times d_a}
\end{align}$$

The goodness for an optimization landscape is the condition number of the objectives' Hessian matrix in optimization theory.

"Effective" condition number of Hessian matrix is defined as 

$$\begin{align}
\kappa_{\mathrm{eff}}(H) := \frac{\sigma_{\max}(H)}{\sigma_{\min}^{+}(H)}
\end{align}$$

This number quantifies how anisotropic the curvature of the objective function is around a point. A large effective condition number indicates that the loss surface has very steep directions alongside very shallow ones (think of a long, narrow valley), which slows down first‑order methods like gradient descent because the step size must be limited by the steepest direction while progress in the flat direction is tiny. Conversely, a condition number close to 1 means the curvature is roughly uniform in all directions, leading to fast, isotropic convergence.

Zero or near‑zero eigenvalues are often ignored because they correspond to flat directions that do not affect convergence, thus we describe "effective" condition number depends on nonzero eigenvalues. 

This Hessian is mathematically tied dierctly to the controllability Gramian of the system. Given that:

1. $z_K$ is affine in $\mathbf{a}$
2. The loss function of the planning problem is quadratic
3. For any matrix $M^T$, the nonzero eigenvalues of $M^TM$ and $MM^T$ coincide.

It can be shown that 

$$\begin{align}
\kappa_{\mathrm{eff}}(H) = \kappa(\mathcal{W}_{K})
\end{align}$$

by

$$\begin{align}
H := \nabla_a^2 \mathcal{L}(a) = 2 J_{\Phi}^{\top} J_{\Phi} \succeq 0
\end{align}$$

where 

$$\begin{align}
J_{\Phi} = \bigl[ A^{K-1} B \;\; A^{K-2} B \;\; \dots \;\; B \bigr] \in \mathbb{R}^{d \times Kd}
\end{align}$$

and 

$$\begin{align}
J_{\Phi}J_{\Phi}^{\top}
\end{align}$$ 

be the finite-horizon controllability Gramian.

$$\begin{align}
\kappa_{\mathrm{eff}}(H) = \kappa(\mathcal{W}_{K}) \le \kappa(B)^{2} \, \frac{\sum_{k=0}^{K-1} \sigma_{\max}(A)^{2k}}{\sum_{k=0}^{K-1} \sigma_{\min}(A)^{2k}} \le \kappa(B)^{2} \, \kappa(A)^{2(K-1)}
\end{align}$$

If the transition is $\epsilon-straight$ with $\epsilon = \lVert A - I \rVert_{2} < 1$, then 

$$\begin{align}
\kappa_{\mathrm{eff}}(H) \le \kappa(B)^{2} \left( \frac{1 + \varepsilon}{1 - \varepsilon} \right)^{2(K-1)}
\end{align}$$

## Better alignment of the Euclidean distance and geodesic distance between two states, in latent space



## Connection to Classical System Identification

While the paper is primarily focused on self-supervised representation learning for world models, its theoretical guarantees are deeply rooted in classical linear systems and control theory. 

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

# My suggestion on future research direction

Why combination of autoregressive frame encoder and causal predictor facilitate straightend dynamic and align Euclidena distance and geodesic distance in latent space? We might get inspiration from the interpretation of MSSA, LISTA-like operator in CRATE for interpretation insight (Manifold flattening and whitebox transformer).

Is there only one unique alignment？ Can we have different encoding but with the same alignment of Euclidean distance and geodesic distance?

Does that mean even the raw data dynamic is already linear, temporal straightening can also improve the optimization efficiency?