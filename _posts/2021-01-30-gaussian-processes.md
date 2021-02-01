---
published: false
layout: post
title: Gaussian Processes 
permalink: /gaussian-processes
---
- [Probabilistic programming](#probabilistic-programming)
- [Bayes rule](#bayes-rule)
- [The naming convention](#the-naming-convention)
- [Gaussian Process](#gaussian-process)
  - [Bivariate Gaussian](#bivariate-gaussian)
- [Posterior predictive distribution](#posterior-predictive-distribution)

There are many ways of building data science models:

* statistical models
* machine learning models
* probabilistic models
 
Probabilistic models use Bayesian statistical methods. 

## Probabilistic programming

So what is probabilistic programming? It's a new **buzz word** of _data science_ that assumes Bayesian inference. 

If you finished your data science classes few years ago, probable you had this topic, and if you studied data science decade ago most likely you haven't had probabilistic programming lectures.

The fact is that everything you can do with the frequentest approach you can do in Bayesian. There is ANOVA in Bayesian, similar for T-test, Logistic regression, everything...


## Bayes rule

We can write:

$\operatorname{P}(\theta \mid x) \propto \prod_{i=1}^{N} \operatorname{P}\left(x_{i} \mid \theta\right) \operatorname{P}(\theta)$

where $\theta$ are unknown parameters, and $x$ are data.

$theta$ can be :
* regression parameters
* hypothesis
* missing data

The inference process says: what do we know about $\theta$ after having observed $x$. If you look the upper formula we also have $P(\theta)$. $P(\theta)$ is what do we know about $\theta$ before we observe our data.

## The naming convention

$\underbrace{\operatorname{P}(\theta \mid x)}_{\text {posterior }} \propto \prod_{i=1}^{N} \overbrace{\operatorname{P}\left(x_{i} \mid \theta\right)}^{\text {data likelihood }} \underbrace{\operatorname{P}(\theta)}_{\text {prior }}$


* Prior distribution: distribution of unknown parametes before we saw the data
* Likelihood principle: All information relevant to the unknown parameters of your model are contained in the likelihood
* Posterior distribution: distribution of unknown parameters after we saw the data $x$

There is no equal sign, the $\propto$ means proportional.

In order to set the equal sign we should write:

$\operatorname{P}(\theta \mid x)=\Large \frac{\prod_{i=1}^{N} \operatorname{P}\left(x_{i} \mid \theta\right) \operatorname{P}(\theta)}{\int_{\theta} \prod_{i=1}^{N} \operatorname{Pr}\left(x_{i} \mid \theta\right) \operatorname{Pr}(\theta) d \theta}$

The integral below you cannot do in closed form except for some special cases. In case we have five parameters $\theta$ this would be integral over five different parameters, and this is hard to compute.

## Gaussian Process

Building flexible non-linear models with Gaussian normal distributions.

Gaussian distribution:

$y \sim N(\mu, \Sigma)$

This doesn't seem like a good idea at first:


### Bivariate Gaussian

$\left.\left.\operatorname{P}\left(y_{1}, y_{2} \mid \mu, \Sigma\right)\right]=\mathcal{N}\left(\left[\begin{array}{c}\mu_{1} \\ \mu_{2}\end{array}\right]\right)\left[\begin{array}{cc}\sigma_{1}^{2} & \sigma_{1} \sigma_{2} \rho \\ \sigma_{1} \sigma_{2} \rho & \sigma_{2}^{2}\end{array}\right]\right)$



In here we have mean vector and covariance matrix.

There are two properties Gaussians have, that are supper neat to create machine learning models:
* they are easy to marginalize


1. marginal distributions of some elements of multivariate normal is also normal

$p(x, y)=\mathcal{N}\left(\left[\begin{array}{l}\mu_{x} \\ \mu_{y}\end{array}\right],\left[\begin{array}{cc}\Sigma_{x} & \Sigma_{x y} \\ \Sigma_{x y}^{T} & \Sigma_{y}\end{array}\right]\right)$


$p(x)=\int p(x, y) d y=\mathcal{N}\left(\mu_{x}, \Sigma_{x}\right)$

...


2. Conditional distribution of some elements of a multivariate normal is also normal


$p(x \mid y)=\mathcal{N}(\underbrace{\mu_{x}+\Sigma_{x y} \Sigma_{y}^{-1}\left(y-\mu_{y}\right)}_{\text {conditional mean }}, \underbrace{\Sigma_{x}-\Sigma_{x y} \Sigma_{y}^{-1} \Sigma_{x y}^{T}}_{\text {conditional covariance }})$


Gaussian process contains infinite elements

$f \sim G P\left(m(x), k\left(x, x^{\prime}\right)\right.$

* $m$ is mean function
* $k$ is covariance function

Function is a generalization of arrays

So what is a Gaussian Process?

**Definition by the book:**

_"An infinite collection of random variables, any finite subset of which have a Gaussian distribution"_.

## Posterior predictive distribution

Answers the question how we predict once we know the model

$\operatorname{Pr}\left(y^{n e w} \mid y\right)=\int \underbrace{\operatorname{Pr}\left(y^{n e w} \mid \theta\right)}_{\text {likelihood }} \overbrace{\operatorname{Pr}(\theta \mid y)}^{\text {posterior }} d \theta$