---
published: false
layout: post
title: Bayesian prerequisites | Algorithms Summary
permalink: /bayesian-algorithms
---


Starting point of any statistical analysis is **the model**.

To create the model mathematicians start from the set of all parameters $\Theta$, where $\theta \in \Theta$. We say parameter $\theta$ is in the set of all parameters.

To define a model we need probability distributions $P_{\theta}$.

 - a collection of probability distributions {Pθ/θ∈Θ} indexed by a parameter θ, where Θ is called the parameter set.



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




### Making Bayesian Learning Tractable

**Shortcuts**

**First shortcut**: Maximum A Posteriori Estimation. simply replace the distribution $P(W \mid \mathcal{S})$ by a Dirac delta function centered on its mode (maximum). 

**Second shortcut**: Maximum Likelihood Estimation. Same as above, but drop the regularizer. Third Shortcut: Restricted Class of function. Simply restrict yourself to special forms of $G(W, Y, X)$ for which the integral can be computed analytically (e.g. Gaussians). CAUTION: This is a perfect example of looking for your lost keys under the street light.

**Fourth shortcut**: Sampling. Draw a a bunch of samples of $W$ from the distribution $P(W \mid \mathcal{S})$, and replace the integral by a sum over those samples.

**Fifth Shortcut**: Local Approximations. compute a Taylor series of $P(W \mid \mathcal{S})$ around its maximum and integrate with the resulting (multivariate) polynomial.


## Sampling methods

### Metropolis Hastings

### Gibs sampling

## Sampling methods shortcomings

These are the possible downsides of sampling:

* we cannot say how fast the sampling method will lead to globally optimal solution
* methods need a good sampling technique to run quickly which may be hard to tell


## Variational inference methods

Sampling methods were invented first (~ 1940) in second world war, but variational inference (invented ~1960) now dominate the field.

Short differences of the two techniques:

* variational approach often cannot find globally optimal solution while sampling can.
* variational approach can converge to the solution up to a certain boundary.
* in practice variational inference methods can use techniques like stochastic gradient optimization and can accelerate for multiple CPUs/GPUs.

