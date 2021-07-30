---
published: false
layout: post
title: Cross Entropy loss calculus | How it works
permalink: /cross-entropy-loss
---

For some time there are some basics I would like to cover. One of the most frequent loss function for classification, both binary and multi label classification. 

Everything start with this image.




>M is positive definite if and only if all of its eigenvalues are positive.

>M {\displaystyle M} M is positive semi-definite if and only if all of its eigenvalues are non-negative.

>M {\displaystyle M} M is negative definite if and only if all of its eigenvalues are negative

> M {\displaystyle M} M is negative semi-definite if and only if all of its eigenvalues are non-positive.

> M {\displaystyle M} M is indefinite if and only if it has both positive and negative eigenvalues.


## Jacobian

The Jacobian $J f$ of a function $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ is the matrix of its first partial derivatives.
$$
J f=\left(\begin{array}{cccc}
\frac{\partial f_{1}}{\partial x_{1}} & \frac{\partial f_{1}}{\partial x_{2}} & \cdots & \frac{\partial f_{1}}{\partial x_{n}} \\
\frac{\partial f_{2}}{\partial x_{1}} & \frac{\partial f_{2}}{\partial x_{2}} & & \frac{\partial f_{2}}{\partial x_{n}} \\
\vdots & & \ddots & \vdots \\
\frac{\partial f_{m}}{\partial x_{1}} & \frac{\partial f_{m}}{\partial x_{2}} & \cdots & \frac{\partial f_{m}}{\partial x_{n}}
\end{array}\right)
$$

We can define Jacobian  at point $p$: $d f_{p}$ as differentiable function $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ at a point $p$ is its best linear approximation at $p$, in the sense that $f(p+h)=f(p)+d f_{p}(h)+o(|h|)$ for small $h$. 

This is the "correct" generalization of the derivative of a function $f: \mathbb{R} \rightarrow \mathbb{R},$ and everything we can do with derivatives we can also do with Jacobians.
In particular, when $n=m$, the determinant of the Jacobian at a point $p$ is the factor by which $f$ locally dilates volumes around $p$ (since $f$ acts locally like the linear transformation $d f_{p},$ which dilates volumes by $\operatorname{det} d f_{p}$ ). This is the reason that the Jacobian appears in the change of variables formula for multivariate integrals, which is perhaps the basic reason to care about the Jacobian. For example this is how one changes an integral in rectangular coordinates to cylindrical or spherical coordinates.

The Jacobian specializes to the most important constructions in multivariable calculus. It immediately specializes to the gradient, for example. When $n=m$ its trace is the divergence. And a more complicated construction gives the curl. The rank of the Jacobian is also an important local invariant of $f$; it roughly measures how "degenerate" or "singular" $f$ is at $p$. This is the reason the Jacobian appears in the statement of the implicit function theorem, which is a fundamental result with applications everywhere.

## Gradient and Hessian 

The gradient $\nabla f$ and Hessian $\nabla^{2} f$ are defined for $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$.


$$
\nabla f=\left(\begin{array}{c}
\frac{\partial f}{\partial x_{1}} \\
\frac{\partial f}{\partial x_{2}} \\
\vdots \\
\frac{\partial f}{\partial x_{n}}
\end{array}\right) \text { and } \nabla^{2} f=\left(\begin{array}{cccc}
\frac{\partial^{2} f}{\partial x_{1}^{2}} & \frac{\partial^{2} f}{\partial x_{1} \partial x_{2}} & \cdots & \frac{\partial^{2} f}{\partial x_{1} \partial x_{n}} \\
\frac{\partial^{2} f}{\partial x_{2} \partial x_{1}} & \frac{\partial^{2} f}{\partial x_{2}^{2}} & & \frac{\partial^{2} f}{\partial x_{2} \partial x_{n}} \\
\vdots & & \ddots & \vdots \\
\frac{\partial^{2} f}{\partial x_{n} \partial x_{1}} & \frac{\partial^{2} f}{\partial x_{n} \partial x_{2}} & \cdots & \frac{\partial^{2} f}{\partial x_{n}^{2}}
\end{array}\right)
$$

First one is a column vector, second one is a matrix.


