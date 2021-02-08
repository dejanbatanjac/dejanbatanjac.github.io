---
published: false
layout: post
title: Bayesian prerequisites
permalink: /bayesian-prerequisites
---

## Probability math

$P(a,b)=P(a|b)P(b)$

Add the new condition:

$P(a,b \mid c)=P(a|b,c)P(b \mid c)$


## MAP

We stated that Bayesian way is dowing the MAP (Maximum a Posteriori) which is a numerical optimization problem.


$\begin{aligned} \theta_{\mathrm{MP}} &=\arg \max _{\theta} P(\theta \mid X)\end{aligned}$

> Note this is a **theta given data** problem

$\begin{aligned}\theta_{\mathrm{MP}} &=\arg \max _{\theta} \frac{P(X \mid \theta) P(\theta)}{P(X)}\end{aligned}$

Note for this problem we don't need the evidence, just the model likelihood which is $P(X \mid \theta)$. 

$\begin{aligned}\theta_{\mathrm{MP}} &=\arg \max _{\theta} P(X \mid \theta) P(\theta) \end{aligned}$


## MAP is **useless**

We tried to avoid computing the evidence to get $\theta$, but there are problems.

Problems:

* If we apply nonlinear function to the input random variable $X \sim \mathcal N$, position of $\theta_{\mathrm MP}$ will change comparing to before the nonlinear function.

> MAP is invalid to reparameterization, except in certain cases where nonlinear function is Gaussian.

* We cannot use the posterior as the input again, so **no online learning**.
* Once we predict the $\theta_{\mathrm MP}$ we are not sure **how confident** we are 
* position for $\theta_{\mathrm MP}$ may not be the indicator of the likelihood distribution. It is frequently **atypical** point (not in the center of the probability density).
> Note likelihood integral may not add to 1, it's not actually a probability

## Conjugate posterior

To overcome the problems with MAP we can pair prior and the likelihood to get the conjugate posterior.

Advantages would be:

* exact posterior 
* easy on-line learning
E.g. $p(\theta \mid X)=B\left(N_{1}+a, N_{0}+b\right)$

Constrains
* Conjugate prior may be inadequate

## Loss functions



Objective function we use:

$\begin{aligned}L(\theta)=\mathbb{I}\left[\theta \neq \theta^{*}\right] \rightarrow \min _{\theta}\end{aligned}$

$\begin{aligned}L(\theta)=\mathbb{E}\left(\theta-\theta^{*}\right)^{2} \rightarrow \min _{\theta}\end{aligned}$

$\begin{aligned}L(\theta)=\mathbb{E}\left|\theta-\theta^{*}\right| \rightarrow \min _{\theta}\end{aligned}$






**Example:** Same distribution

Imagine that you are training a neural network to play games. X is an image of the game screen and $\thetaθ$ are network parameters. 

If we knew $P(X)$, we could generate new game-like frames.



## Conjugate prior for a likelihood function

Prior is said to be **conjugate for a likelihood** function if the posterior would stay in the same family of distributions as prior.

Why conjugate priors are useful? 

**There is no need for computing integrals because we get posterior in explicit form.**

Finding a conjugate prior is useful because:

If posterior stays in the same family with prior, the integral $p\left(x_{n e w} \mid x\right)=\int p\left(x_{n e w} \mid \theta\right) p(\theta \mid x) d \theta$ for prediction is also tractable.

This integral is called **the evidence** and it can be computed analytically if prior, likelihood and posterior are known.

> We can perform analytical inference and find posterior distribution instead of taking point MAP estimate



Matches:

## Beta prior, binomial likelihood

$$
\begin{array}{|c|c|c|c|c|}
\hline \text { hypothesis } & \text { data } & \text { prior } & \text { likelihood } & \text { posterior } \\
\hline \theta & x & \text { beta }(a, b) & \text { binomial }(N, \theta) & \text { beta }(a+x, b+N-x) \\
\hline \theta & x & c_{1} \theta^{a-1}(1-\theta)^{b-1} & c_{2} \theta^{x}(1-\theta)^{N-x} & c_{3} \theta^{a+x-1}(1-\theta)^{b+N-x-1} \\
\hline
\end{array}
$$


## Beta prior bernoulli likelihood

$$
\begin{array}{|c|c|c|c|c|}
\hline \text { hypothesis } & \text { data } & \text { prior } & \text { likelihood } & \text { posterior } \\
\hline \theta & x & \text { beta }(a, b) & \text { Bernoulli }(\theta) & \text { beta }(a+1, b) \text { or beta }(a, b+1) \\
\hline \theta & x=1 & c_{1} \theta^{a-1}(1-\theta)^{b-1} & \theta & c_{3} \theta^{a}(1-\theta)^{b-1} \\
\hline \theta & x=0 & c_{1} \theta^{a-1}(1-\theta)^{b-1} & 1-\theta & c_{3} \theta^{a-1}(1-\theta)^{b} \\
\hline
\end{array}
$$

## Beta prior, geometric likelihood

$$
\begin{array}{|c|c|c|c|c|}
\hline \text { hypothesis } & \text { data } & \text { prior } & \text { likelihood } & \text { posterior } \\
\hline \theta & x & \text { beta }(a, b) & \text { geometric }(\theta) & \text { beta }(a+x, b+1) \\
\hline \theta & x & c_{1} \theta^{a-1}(1-\theta)^{b-1} & \theta^{x}(1-\theta) & c_{3} \theta^{a+x-1}(1-\theta)^{b} \\
\hline
\end{array}
$$


## Normal prior normal likelihood

$$
\begin{array}{|c|c|c|c|c|}
\hline \text { hypothesis } & \text { data } & \text { prior } & \text { likelihood } & \text { posterior } \\
\hline \theta & x & f(\theta) \sim \mathrm{N}\left(\mu_{\text {prior }}, \sigma_{\text {prior }}^{2}\right) & f(x \mid \theta) \sim \mathrm{N}\left(\theta, \sigma^{2}\right) & f(\theta \mid x) \sim \mathrm{N}\left(\mu_{\text {post }}, \sigma_{\text {post }}^{2}\right) \\
\hline \theta & x & c_{1} \exp \left(\frac{-\left(\theta-\mu_{\text {prior }}\right)^{2}}{2 \sigma_{\text {prior }}^{2}}\right) & c_{2} \exp \left(\frac{-(x-\theta)^{2}}{2 \sigma^{2}}\right) & c_{3} \exp \left(\frac{-\left(\theta-\mu_{\text {post }}\right)^{2}}{2 \sigma_{\text {post }}^{2}}\right) \\
\hline
\end{array}
$$

