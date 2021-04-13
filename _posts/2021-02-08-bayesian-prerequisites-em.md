---
published: false
layout: post
title: Bayesian prerequisites | Algorithms Summary
permalink: /bayesian-algorithms
---
- [EM algorithm](#em-algorithm)
  - [M-step of EM](#m-step-of-em)


## EM algorithm

In EM algorithm, we maximize variational lower bound $\mathcal{L}(q, \theta)=\log p(X \mid \theta)-\mathrm{KL}(q \| p)$ with respect to $q$ (E-step) and $\theta$ (M-step) iteratively. Why is the maximization of lower bound on E-step equivalent to minimization of $\mathrm{KL}$ divergence?

Because uncomplete likelihood does not depend on $q(Z)$
Because we cannot maximize lower bound w.r.t. $q(Z)$
Because posterior becomes tractable
Because of Jensen's inequality


https://www.robots.ox.ac.uk/~mosb/teaching/B14/5_slides_MAP.pdf


>Bayesian theory is subjective. This description is often used in a pejorative sense by frequentists!

Bayesians see it as desirable to explicitly acknowledge all relevant prior information.

If the predictions don't match our expectations, we've learned something.

We can use what we've learned to revise and improve our prior models.

Maximum likelihood is unable to benefit fromtheinformation contained in our priors.Maximum likelihood effectively ignores the prior.Often our priors actually contain much useful informationour inference can exploit.







### M-step of EM

In Expectation Maximization algorithm we try to maximize the expected value of logarithm of joint probability distribution conditioned on parameters.

$X$ is our input, $T$ latent variables and posterior distribution is $q$ and $\theta$ the parameters.

$$
\max _{\theta} \mathbb{E}_{q} \log p(X, T \mid \theta)
$$

The distribution we sample from is:
$$
p(X, T \mid \theta)
$$


