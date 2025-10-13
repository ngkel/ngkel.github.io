---
layout: distill
title: Optimization from basics to ISTA
description: 
tags: optimization
giscus_comments: true
date: 2025-08-28
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
  - name: Introduction to ISTA
  - name: Convex Functions
  - name: Gradient Descent
  - name: Subgradient Method
  - name: Proximal Gradient Descent
  - name: How solution to LASSO inspire more efficient solution to basis pursuit
  - name: Connection between ISTA and deep network

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

# Introduction to ISTA

This is a blog for reviewing important optimization concepts that are critical for understanding ISTA. ISTA is an algorithm that applies proximal gradient to the LASSO objective function when solving sparse coding problem give an overcomplete dictionary. The objective function of the sparse coding with an overcomplete dictionary is as follow:

$$\begin{align}
\min_{\mathbf{Z} \in \mathbb{R}^{d \times N}} \left\{ \left\| \mathbf{X} - \mathbf{A}\mathbf{Z} \right\|_F^2 + \lambda \|\mathbf{Z}\|_1 \right\}
\end{align}$$

where Frobenius Norm is defined as:

$$\begin{align}
\|\mathbf{A}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n \left| a_{ij} \right|^2}
\end{align}$$

# Convex Functions

A differentiable function $\mathcal{f}: \mathcal{D} \rightarrow \mathbb{R}$ is convex if and only if 

$$\begin{align}
f(x') \geq f(x) + \nabla f(x)^*(x' - x), \forall \ x, x' \in \mathcal{D}.
\end{align}$$


where $\mathcal{D}$ is a convex set.

This means the graph of $f$ lies below or on the line segment joining any two points on the graph (the "chord" between them). Geometrically, convex functions have a "U-shaped" or "bowl-shaped" graph—no "dents" or local maxima that trap you from reaching the global optima.


# Gradient Descent
The well-known formula for gradient descent is:

$$\begin{align}
x_{k+1} = x_k - h \, \nabla f(x_k)
\end{align}$$

In gradient descent, if we find the right step size, the change of the value of the objective function has the following properties:

1. In convex optimization, gradient descent improves at every iteration given correct step size.
2. G.D. has self-tuning property - It takes bigger step size when it is far away from the optimal point. Vice versa. And such self-tuning property does not depend on the error at each iteration. For example, each update step of function $3x^2 +4x -2$ is $x \gets x - h (6x + 4) $ The gradient ($6x + 4$) is big when it is far away from the optimal point and smaller when it is closed. The self-tuning property is always inplace with a fixed step size, but we need to make sure the step size is chosen properly.

These are just some useful properties for convex optimization. Things discussed below are not limited to convex optimization.

The challenges of G.D. are the following:

1. How should we determine the size $h$ such that it converges, and more importantly converges faster?
2. What if the function is not differentiable at some point?

## Proof of guarantee of improvement in each gradient step for $\beta$-smooth function with step size being $\frac{1}{\beta}$

The challenge number 2 will be addressed in the next section. For the rest of this section, we will explain how step size is chosen for smooth, Lipschitz continuous function with Lipschitz constant $\beta$. Smooth function is not neccessarily convex thus we should not assume the property we are discussing here for any $\beta$-smooth function is only applicable to convex function. 

We end this section with a caveat: learning a global optimum is (usually) impractically hard. Under certain conditions, we can ensure that the gradient descent iterates converge to a local optimum. Also, under more relaxed conditions, we can ensure local convergence, i.e., that the iterates converge to a (global or local) optimum if the sequence is initialized close enough to the optimum.

Recall that a differentiable function $f(x)$ has L-Lipschitz continuous gradients if 

$$\begin{align}
\|\nabla f(y) - \nabla f(x)\| \leq \beta \|y - x\| \quad \forall x, y
\end{align}$$

**Lemma.** Suppose that $f$ is differentiable and $\nabla f$ is $\beta$-Lipschitz. Then for every $x, x' \in \mathbb{R}^n$,

$$\begin{align}
f(x') \leq \hat{f}(x', x) = f(x) + \langle \nabla f(x), x' - x \rangle + \frac{\beta}{2} \|x' - x\|_2^2  \\

\quad\quad\;\;\; = \frac{\beta}{2} \left\| x' - \left(x - \frac{1}{\beta} \nabla f(x)\right) \right\|_2^2 + h(x).
\end{align}$$

for some $h(x)$ that does not depend on $x'$ while $x$ is the base point. This baiscally means that we can always fit a quadratic upper bound on every point $x'$ of a Lipschitz continuous function that is $\beta$-smooth.

We denote this upper bound as $\hat{f}(x', x)$. The minimizer of this upper bound is

$$\begin{align}
\arg \min_{x'} \hat{f}(x', x) = x - \frac{1}{\beta} \nabla f(x)
\end{align}$$

This is exactly in the form of a gradient step in gradient descent. And because 
And because

$$\begin{align}
f(x) = \hat{f}(x, x)
\end{align}$$

And

$$\begin{align}
\hat{f}(x'_{*}, x) \leq \hat{f}(x, x)
\end{align}$$

Thus,

$$
\begin{align}
f(x'_*) \leq \hat{f}(x'_*, x) \leq \hat{f}(x, x) = f(x)
\end{align}
$$

Which imply that if we apply the gradient descent method with step size 1/$L$, we are guaranteed to produce the monotone sequence of function values $f(x_{k})$.


# Subgradient Method
To cope with the challenge number 2, we might consider subgradient method. Let's consider the example problem:

$$\begin{align}
f(x) = |x|
\end{align}$$

At $x = 0$, unlike G.D., the gradient is not uniquely defined. Thus, we replace the concept gradient with another fancy term called subdifferential, which is actually a generalization of gradient. The definition of subdifferential is as follow:

For a convex function $ f: \mathbb{R}^n \to \mathbb{R} $ the subdifferential $ \partial f(x) $ at a point $ x $ is the set of all vectors $ g \in \mathbb{R}^n $ that satisfy the subgradient inequality:

$$\begin{align}
f(y) \geq f(x) + g^\top (y - x) \quad \forall y \in \mathbb{R}^n.
\end{align}$$

Note that the subgradient inequality does not generally hold for non-convex problem. The key idea of subgradient method is then, for each update we do the same as G.D. except if we touch the non-differentiable point, we randomly pick a subdifferential in the set of valid subdifferentials. As the value oscillate around the non-differentiable point, we might apply some mechanism to shrink the step size. We repeat the iteration until it converges.

The subgradient method algorithm can be summarized as follow：

The subgradient method iteratively updates the iterate $z_k$ toward the minimum:

$$\begin{align}
x_{k+1} = x_k - h_k g_k
\end{align}$$

where:

- $g_k$ is a subgradient at $z_k$.

- $h_k > 0$ is the step size at iteration $k$.

Step Size Choices:

- Fixed step size ($h_k = h$, $h = \frac{b}{2}$) may cause oscillation near the optimum; useful for illustration.

- Diminishing step size (e.g., $h_k = \frac{a}{k}$) for constants $a > 0$, $b \geq 0$: Ensures convergence for bounded subgradients.

## Optimality condition of subgradient methods
$$
\text{Theorem.} \quad \text{Suppose that } f: \mathbb{R}^d \to \mathbb{R} \cup \{\infty\} \text{ is a convex function. Then, } x^* \text{ minimizes } f(x) \text{ if and only if } 0 \in \partial f(x^*)
$$

## Convergence rate of subgradient method
The main disadvantage to this approach is its relatively poor convergence rate.

Let $x_*$ be the (global) minimizer of $f(x)$. In general, the convergence rate of the subgradient method for nonsmooth objective functions, in terms of the function value $f(x_k) - f(x_*)$, is:

$$
O(1/\sqrt{k}).
$$

The constants in the big-$O$ notation depend on various properties of the problem. The important point is that for even a moderate target accuracy
$$\begin{align}
f(x_k) - f(x_*) \leq \epsilon,
\end{align}$$
we will have to set $k = O(\epsilon^{-2})$ very large.


# Proximal Gradient Descent
Most objective functions such as that of ISTA problem is a composite of smooth and non-smooth functions.

$$\begin{align}
F(x) = f(x)+g(x)
\end{align}$$

where $f(x)$ is a $\beta$-smooth function and $g(x)$ is non-differentiable function such as $l_1$ norm. Therefore, the composite function $F(x)$ is not differentiable thus G.D. does not apply. The first recourse in this situation is to replace the gradient with a subgradient. The main disadvantage to this approach is its relatively poor convergence rate. Recall that thanks to self-tuning property of gradient descent method for smooth function, the convergence rate of G.D. is much better (at a rate of $O(1/k)$) than subgradient method. Can we draw inspiration from the gradient method to produce a more efficient algorithm for minimizing the composite function such that we can avoid slow convergence due to $x_{k+1} = x_k - h_k g_k$?

As before, for the composite objective $F(x) = f(x) + g(x)$, we can construct an upper quadratic bound of $F(x)$ at $x_k$ using the smoothness of $f(x)$:

$$
\hat{F}(x, x_k) = \frac{\beta}{2} \left\| x - \left( x_k - \frac{1}{\beta} \nabla f(x_k) \right) \right\|_2^2 + h(x_k) + g(x),
$$

$$
\quad\quad\;\;\; = \hat{f}(x, x_k) + g(x).
$$

where $\beta$ is the Lipschitz constant of the gradient of $f(x)$. Notice that $h(x_k)$ collects terms independent of $x$ (absorbing everything not affecting the minimization). This upper bound motivates the **proximal gradient step**:

$$
\begin{align}
x_{k+1} &= \arg\min_x\, \frac{\beta}{2} \left\| x - \left( x_k - \frac{1}{\beta} \nabla f(x_k) \right) \right\|_2^2 + g(x) \\
        &= \arg\min_x\, \frac{\beta}{2} \left\| x - w_k \right\|_2^2 + g(x) ,
\end{align}
$$
where $w_k = x_k - \frac{1}{\beta} \nabla f(x_k)$ which only depends on $x_k$. 

The minimization above is exactly the **proximal operator** of $g$ applied at the "gradient descent step" for $f$:

$$
x_{k+1} = \mathrm{prox}_{g, \frac{1}{\beta}}\left( x_k - \frac{1}{\beta} \nabla f(x_k) \right) = \mathrm{prox}_{g, \frac{1}{\beta}}\left(w_k \right) = \arg\min_x\, \frac{\beta}{2} \left\| x - w_k \right\|_2^2 + g(x)
$$

This combines the fast convergence of gradient descent for the smooth part with the regularization/control of the non-smooth part via the proximal operator. The resulting algorithm enjoys the same O(1/k) convergence rate as in the smooth case. Recognizing special structure in our problem of interest yields a significantly more accurate and eﬃcient algorithm. In fact, we can further apply acceleration method similar to acceleration method for purely smooth function to speed up the convergence! This is also the key reason how people may come up with Adam, one of the most popular optimizer in deep learning. We may talk about this in the future blog. 

## Solving LASSO by proximal gradient

Let's see how proximal operator looks like for LASSO problem:

$$
\mathrm{prox}_{g, \frac{1}{\beta}}\left(w_k \right) = \arg\min_x\, \frac{\beta}{2} \left\| x - w_k \right\|_2^2 + g(x)
$$

We can obtain the argument of the minimum by setting:

$$
0 \in (x - w) + \lambda \partial \|x\|_1
$$

where $w = w_k$, $\lambda = 1/\beta$, and $\partial \|x\|_1$ denotes the subdifferential of the $\ell_1$ norm. For each coordinate $i = 1, \ldots, n$:

$$
0 \in (x_i - w_i) + \lambda \partial |x_i| =
\begin{cases}
x_i - w_i + \lambda, & x_i > 0 \\
-w_i + \lambda[-1, 1], & x_i = 0 \\
x_i - w_i - \lambda, & x_i < 0
\end{cases}
$$
Therefore, the solution to this optimality condition is the **soft-thresholding function applied element-wise**:
$$
x_i^* = \mathrm{soft}(w_i, \lambda) \triangleq \mathrm{sign}(w_i) \max(|w_i| - \lambda, 0) \qquad i = 1, \ldots, n.
$$

The pseudocode of PGD for LASSO problem is as follow:

{% include figure.liquid path="assets/img/pgd_for_lasso.png" class="img-fluid rounded z-depth-1" %}

As the iteration goes on, more and more value in the solution vector may go zero to promote sparsity. The soft-threshold is determined by $\frac{\lambda}{\beta}$ (replace $L$ by $\beta$ in the picture)  

## Similarity of proximal operator and nonlinear function in deep network

The soft-thresholding in LISTA looks notably similar to ReLU in deep network.

# How solution to LASSO inspire more efficient solution to basis pursuit
When I first learn about these 2 problems, most textbook would introduce them as following roughly: LASSO is a variant of the BP, considering the noisy case in which the original signal is contaminated by moderate Gaussian noise $\mathcal{z}$. 

Basis Pursuit:

$$
\begin{aligned}
&\min_{x} \quad && \|x\|_1 \\
&\text{subject to} \quad && Ax = y.
\end{aligned}
$$

LASSO:

$$\begin{align}
\min_{x} \ \frac{1}{2} \|y - Ax\|_2^2 + \lambda \|x\|_1
\end{align}$$

where the observations $y$ are generated as $y = Ax_0 + z$, for some unknown sparse $x_0$ and noise $z$.

However, after I went through the optimization textbook or chapters in related textbook over and over again, I have found that there is a deeper and valuable insight. First of all, let's recall the core difference between 2 problems. BP or BPDN enforce equality or inequality constraints while LASSO does not strictly do so. In LASSO, it is up to us to control the $\lambda$. 

Because of the requirement to enforce equality constraint on BP, the first intuitive method comes to people mind is to apply projected subgradient method to solve BP. The key idea of BP is to alternate between subgradient steps and projection operation. The convergence rate thus is mainly determined by subgradient step for non-smooth and non-strongly convex $\|x\|_1$, given by:

$$
O(1/\sqrt{k}).
$$

The constants in the big-$O$ notation depend on various properties of the problem. The important point is that for even a moderate target accuracy
$$\begin{align}
f(x_k) - f(x_*) \leq \epsilon,
\end{align}$$
we will have to set $k = O(\epsilon^{-2})$ very large.

Apart from using projected subgradient method to solve this problem, another approach is to solve a series of problems of this type :

$$\begin{align}
\min_x \ \|x\|_1 + \frac{\mu}{2} \|y - Ax\|_2^2
\end{align}$$

for an increasing sequence of $\mu_i \to \infty$ to enforce the equality. 

Now, the deeper insight I have mentioned will be explained starting from here. 

First of all, it is obvious that for each $\mu_i$, the problem is exactly a LASSO problem, which can be solved by proximal gradient descent (PGD). 

PGD, unlike projected subgradient descent, benefit from the smooth part $\|y - Ax\|_2^2$ and thus the convergence rate is $O(1/k)$. It also allows us to utilize acceleration techniques
that were designed for smooth functions and lead to much more scalable and fast-converging algorithms, with convergence rates much better than the generic situation. Therefore, in each iteration for each $\mu_i$. we solve the problem more efficiently. However, as the weight $\mu_i$ increases, the corresponding BPDN problem becomes increasingly ill-conditioned and hence algorithms converge slower. This is because  for first-order methods such as PG and APG, the rate of convergence is dictated by how quickly the gradient $\nabla(\mu f) = \mu A^T (A x - y)$ can change from point-to-point, which is measured through the Lipschitz constant. 

$$\begin{align}
L_{\nabla_{\mu}f} = \mu \|\mathbf{A}\|_2^2.
\end{align}$$

This increases linearly with $\mu$: The larger $\mu$ is, the harder the unconstrained problem is to solve! We can then employ the augmented Lagrange multiplier (ALM) technique to
alleviate this diﬃculty.

# Connection between ISTA and deep network
Recall that in each layer of forward pass of feature transformation in ISTA is given by the following process:

$$\begin{align}
\mathbf{Z}_{\ell+1} = S_{\eta, \lambda} \left( \mathbf{Z}_\ell - 2\eta (\mathbf{A}_\ell)^\top \left( \mathbf{A}_\ell \mathbf{Z}_\ell - \mathbf{X} \right) \right),\quad \forall \ell \in [L].
\end{align}$$

It looks like the forward pass of a deep neural network with weights given by $\mathbf{A}$. The first part of this blog



