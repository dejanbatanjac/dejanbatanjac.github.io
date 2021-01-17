---
published: true
layout: post
title: Expectation
permalink: /expectation
---

In here I will set some notation of the mathematical expectation of discrete and continuous random variable (RV)

### Discrete RV expectation

In case of the discrete variable the expectation, or expected value, of some function $f(x)$ with respect to a probability distribution $P(x)$ is the average, or mean value, that $f$ takes on when $x$ is drawn from $P$:


$\begin{aligned} \mathbb{E}_{\mathrm{x} \sim P} [ f(x) ]= \sum_{x} P(x) f(x) \end{aligned}$


$\mathrm{x} \sim P$ means $\mathrm{x}$ is drawn from distribution $P(x)$ or just from $P$. Inside the $[\ldots]$ brackets we should have some function $f(x)$, or in special case just $x$.

### Continuous RV expectation

For continuous variables it is computed with the integral:

$\mathbb{E}_{\mathrm{x} \sim p}[f(x)]=\int p(x) f(x) d x$


If the identity of the distribution is clear from the context we may write simple:

$\mathbb{E}_{\mathrm{x} }[f(x)]$

If the random variable is clear from the context we may write:


$\mathbb{E}[f(x)]$


By default, we can assume that

$\mathbb{E}[\cdot]$

averages over the values of all the random variables inside the brackets. 

Likewise, when there is no ambiguity, we may omit the square brackets:

$\mathbb{E}$

Expectations are linear:


$\mathbb{E} \_ {\mathrm{x}}[\alpha f(x)+\beta g(x)]=\alpha \mathbb{E}_{\mathrm{x}}[f(x)]+\beta \mathbb{E}_{\mathrm{x}}[g(x)]$



We define the <span> $\mathbb X = \{{\pmb x^{(1)}}, \ldots ,{\pmb x^{(m)}}\}$ </span>

$p_{data}(\mathrm x)$

$p_{model}(\pmb {\mathrm x}; \pmb \theta)$


$p_{model}(\pmb {x}; \pmb \theta)$ maps any concrete configuration to $p_{data}(\pmb {x})$

$\theta_{ML} = arg max$


$\begin{aligned} \boldsymbol{\theta}_{\mathrm{ML}} &=\underset{\boldsymbol{\theta}}{\arg \max } p_{\text {model }}(\mathbb{X} ; \boldsymbol{\theta}) \\ &=\underset{\boldsymbol{\theta}}{\arg \max } \prod_{i=1}^{m} p_{\text {model }}\left(\boldsymbol{x}^{(i)} ; \boldsymbol{\theta}\right) \end{aligned}$


For numeric stability:

$\boldsymbol{\theta}_{\mathrm{ML}}=\underset{\boldsymbol{\theta}}{\arg \max } \sum_{i=1}^{m} \log p_{\text {model }}\left(\boldsymbol{x}^{(i)} ; \boldsymbol{\theta}\right)$

Defined by train data:

$\boldsymbol{\theta}_{\mathrm{ML}}=\underset{\boldsymbol{\theta}}{\arg \max } \mathbb{E}_{\mathbf{x} \sim \hat{p}_{\text {data }}} \log p_{\text {model }}(\boldsymbol{x} ; \boldsymbol{\theta})$

Final:

$D_{\mathrm{KL}}\left(\hat{p}_{\text {data }} \| p_{\text {model }}\right)=
\mathbb{E}_{\mathbf{x} \sim \hat{p}_{\text {data }}}
\left[\overbrace{\log \hat{p}_{\text {data }}(\pmb{x})}^{\ data \ generating \ process}-\log p_{\text {model }}(\pmb{x})\right]$

Important part: 

$\mathbb{E}_{\mathbf{x} \sim \hat{p}_{\text {data }}}\left[-\log p_{\text {model }}(\pmb{x})\right]$

where:

$\hat{p}_{\text {data }}$ is empirical distribution, ${p}_{\text {data }}$ is true distribution

Once we have the expectation we can define the variance and covariance.

## Variance


Variance give us the measure how much values of a function of a random variable $\text x$ vary as we sample different values of $\mathrm x$ from it's probability distribution.


$\operatorname{Var}(f(x))=\mathbb{E}\left[(f(x)-\mathbb{E}[f(x)])^{2}\right]$




## Covariance

The covariance gives some sense of how much two values are linearly related to each other, as well as the scale of these variables:

$\operatorname{Cov}\left(f(x), g(y)\right)=\mathbb{E}[(f(x)-\mathbb{E}[f(x)])(g(y)-\mathbb{E}[g(y)])]$