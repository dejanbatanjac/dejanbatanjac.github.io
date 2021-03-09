---
published: false
layout: post
title: Bayesian Prerequisites | Random Variables
permalink: /probability-models
---
- [Discrete and continuous Random Variables](#discrete-and-continuous-random-variables)
- [Probability rules](#probability-rules)
  - [Joint rule:](#joint-rule)
  - [Sum rule:](#sum-rule)
  - [Bayes rule](#bayes-rule)
- [Combine two random variables](#combine-two-random-variables)
  - [Examples](#examples)
- [Marginal likelihood](#marginal-likelihood)

## Discrete and continuous Random Variables

We know discrete and continuos random variables. 

An output from a dice is discrete: 1,2,3,4,5,6.

A non diabetic blood sugar can take continuos values from 4.0 to 5.9 mmol/L before the lunch

For discrete RV we define PMF (probability mass function)

$P(X)=\left\{\begin{array}{cc}0.2 & X=1 \\ 0.5 & X=3 \\ 0.3 & X=7 \\ 0 & \text { otherwise }\end{array}\right.$


For continuos RV we define PDF (probability density 
function)

$\begin{aligned}\int_{-\infty}^{\infty} p(x) \mathrm{d} x =1\end{aligned}$

> PDF should add to 1

We can measure the probability on a interval:

$P(x \in[a, b])=\int_{a}^{b} p(x) d x < 1$

> It should always be less then 1

## Probability rules

### Joint rule:



### Sum rule:

$p(X)=\int_{-\infty}^{\infty} p(X, Y) d Y$

When you know the joint probability you can integrate the RV $Y$ to get the $P(X)$. This is also called marginalization.

### Bayes rule

$P(\theta \mid X) \Large = \frac{P(X, \theta)}{P(X)}=\frac{P(X \mid \theta) P(\theta)}{P(X)}$

Where: 
* $\theta$ are parameters
* $X$ observations
* $P(X)$ the evidence
* $P(X \mid \theta)$ Likelihood (how well parameters explain our data)
* $P(\theta)$ Prior
* $P(\theta \mid X)$ Posterior (probability of parameters after we observer the data)


## Combine two random variables

Having $X$ and $Y$ we can create sum random variable $Z$

$\begin{array}{l}
X \sim N\left(\mu_{X}, \sigma_{X}^{2}\right) \\
Y \sim N\left(\mu_{Y}, \sigma_{Y}^{2}\right) \\
Z=X+Y,
\end{array}$

then

$Z \sim N\left(\mu_{X}+\mu_{Y}, \sigma_{X}^{2}+\sigma_{Y}^{2}\right)$

Case we need $Z=X-Y$ this would be similar.

$Z \sim N\left(\mu_{X}-\mu_{Y}, \sigma_{X}^{2}+\sigma_{Y}^{2}\right)$


### Examples


## Marginal likelihood

Given a set of independent identically distributed data points $\mathbf{X}=\left(x_{1}, \ldots, x_{n}\right),$ where $x_{i} \sim p\left(x_{i} \mid \theta\right)$ according to some probability distribution parameterized by $\theta$, where $\theta$ itself is a random variable described by a distribution, i.e. $\theta \sim p(\theta \mid \alpha),$ the marginal likelihood in general asks what the probability $p(\mathbf{X} \mid \alpha)$ is, where $\theta$ has been marginalized out (integrated out):
$$
p(\mathbf{X} \mid \alpha)=\int_{\theta} p(\mathbf{X} \mid \theta) p(\theta \mid \alpha) \mathrm{d} \theta
$$
The above definition is phrased in the context of Bayesian statistics. In classical (frequentist) statistics, the concept of marginal likelihood occurs instead in the context of a joint parameter $\theta=(\psi, \lambda),$ where $\psi$ is the actual parameter of interest, and $\lambda$ is a non-interesting nuisance parameter. If there exists a probability distribution for $\lambda,$ it is often desirable to consider the likelihood function only in terms of $\psi$, by marginalizing out $\lambda$ :
$$
\mathcal{L}(\psi ; \mathbf{X})=p(\mathbf{X} \mid \psi)=\int_{\lambda} p(\mathbf{X} \mid \lambda, \psi) p(\lambda \mid \psi) \mathrm{d} \lambda
$$
Unfortunately, marginal likelihoods are generally difficult to compute. Exact solutions are known for a small class of distributions, particularly when the marginalized-out
parameter is the conjugate prior of the distribution of the data. In other cases, some kind of numerical integration method is needed, either a general method such as Gaussian integration or a Monte Carlo method, or a method specialized to statistical problems such as the Laplace approximation, Gibbs/Metropolis sampling, or the EM algorithm.

It is also possible to apply the above considerations to a single random variable (data point) $x$, rather than a set of observations. In a Bayesian context, this is equivalent to the prior predictive distribution of a data point.
