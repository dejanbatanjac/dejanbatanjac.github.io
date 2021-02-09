---
published: false
layout: post
title: Bayesian prerequisites | MLE
permalink: /bayesian-mle
---


https://www.robots.ox.ac.uk/~mosb/teaching/B14/5_slides_MAP.pdf


Bayesian theory is subjective. This description is often used in a pejorative sense by frequentists!

Bayesians see it as desirable to explicitly acknowledge all relevant prior information.

If the predictions don't match our expectations, we've learned something.

We can use what we’ve learned to revise and improve our prior models.

Maximum likelihood is unable to benefit fromtheinformation contained in our priors.Maximum likelihood effectively ignores the prior.Often our priors actually contain much useful informationour inference can exploit.

## Le Cun

https://cs.nyu.edu/%7Eyann/2007f-G22-2565-001/diglib/lecture07-bayes.pdf

### Making Bayesian Learning Tractable

**Shortcuts**

**First shortcut**: Maximum A Posteriori Estimation. simply replace the distribution $P(W \mid \mathcal{S})$ by a Dirac delta function centered on its mode (maximum). 

**Second shortcut**: Maximum Likelihood Estimation. Same as above, but drop the regularizer. Third Shortcut: Restricted Class of function. Simply restrict yourself to special forms of $G(W, Y, X)$ for which the integral can be computed analytically (e.g. Gaussians). CAUTION: This is a perfect example of looking for your lost keys under the street light.

**Fourth shortcut**: Sampling. Draw a a bunch of samples of $W$ from the distribution $P(W \mid \mathcal{S})$, and replace the integral by a sum over those samples.

**Fifth Shortcut**: Local Approximations. compute a Taylor series of $P(W \mid \mathcal{S})$ around its maximum and integrate with the resulting (multivariate) polynomial.

