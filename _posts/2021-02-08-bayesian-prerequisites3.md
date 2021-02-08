---
published: false
layout: post
title: Bayesian prerequisites | EM
permalink: /bayesian-expectation-minimization
---

## Bayes rule

Computing posterior is know as inference problem.

$P(\theta \mid X) \Large = \frac{P(X, \theta)}{P(X)}=\frac{P(X \mid \theta) P(\theta)}{P(X)}$

Where: 
* $\theta$ are parameters
* $X$ observations
* $P(X)$ the evidence
* $P(X \mid \theta)$ Likelihood (how well parameters explain our data)
* $P(\theta)$ Prior
* $P(\theta \mid X)$ Posterior (probability of parameters after we observer the data)

## Inference

$P(X)=\int P(X, \theta) d \theta$


## EM algorithm

In EM algorithm, we maximize variational lower bound $\mathcal{L}(q, \theta)=\log p(X \mid \theta)-\mathrm{KL}(q \| p)$ with respect to $q$ (E-step) and $\theta$ (M-step) iteratively. Why is the maximization of lower bound on E-step equivalent to minimization of $\mathrm{KL}$ divergence?

Because uncomplete likelihood does not depend on $q(Z)$
Because we cannot maximize lower bound w.r.t. $q(Z)$
Because posterior becomes tractable
Because of Jensen's inequality