---
published: true
layout: post
title: Bayesian prerequisites | Beta and Gamma distributions
permalink: /bayesian-prerequisites-beta-gamma
---
- [Beta distribution](#beta-distribution)
- [Gamma distribution](#gamma-distribution)


In here I will create short intro for beta and gamma distributions two very important distributions form [different univariate distributions](http://www.math.wm.edu/~leemis/chart/UDR/UDR.html).

## Beta distribution


The beta distribution has support over the interval [0,1] and is defined as follows:
$$
\operatorname{Beta}(x \mid a, b)=\frac{1}{B(a, b)} x^{a-1}(1-x)^{b-1}
$$
Where $B(p, q)$ is the beta function:
$$
B(a, b) = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a+b)}
$$


Beta function $B$ can also be defined as:

$B(a, b)=\int_{0}^{1} u^{a-1}(1-u)^{b-1} d u \quad a, b \in(0, \infty)$


Parameters:
* $a$ left parameter
* $b$ right parameter


If $a, b \in(0, \infty)$ then
$B(a, b)=\frac{\Gamma(a) \Gamma(b)}{\Gamma(a+b)}$

Special case: Incomplete beta function:

$B(x ; a, b)=\int_{0}^{x} u^{a-1}(1-u)^{b-1} d u, \quad x \in(0,1) ; a, b \in(0, \infty)$

When $x=1$, this is complete beta function.


**Beta distribution (PDF)**:

> To distinguish $B(a,b)$ is not same as $Beta(a, b) =B(x \mid a,b)$

$B(x \mid a,b)=\frac{1}{B(a, b)} x^{a-1}(1-x)^{b-1}, \quad x \in(0,1)$

where parameters $a$ and $b$ need to be positive.

The **mean** and **variance** of **beta distribution**:



$\mathbb{E}[X]=\Large \frac{a}{a+b}$

$\mathbb{V}[X]=\Large \frac{a b}{(a+b)^{2}(a+b-1)}$

$\operatorname{Mode}[X]=\Large \frac{a-1}{a+b-2}$



> Beta distribution is conjugate to the Bernoulli likelihood.

$p(X \mid \theta)=\theta^{N_{1}}(1-\theta)^{N_{0}}$
$p(\theta)=B(\theta \mid a, b) \propto \theta^{a-1}(1-\theta)^{b-1}$
$p(\theta \mid X) \propto p(X \mid \theta) p(\theta)$
$p(\theta \mid X) \propto \theta^{N_{1}}(1-\theta)^{N_{0}} \cdot \theta^{a-1}(1-\theta)^{b-1}$
$p(\theta \mid X) \propto \theta^{N_{1}+a-1}(1-\theta)^{N_{0}+b-1}$
$p(\theta \mid X)=B\left(N_{1}+a, N_{0}+b\right)$

---

**Example**: _Percentage of pdf inside interval_

Let's calculate $Beta(3,1)$ inside $[0,0.5]$ interval using R code.

The beta function in R can be implemented using the `beta(a,b)` however to integrate the pdf we need `dbeta` function.


```R
integrate(function(p) dbeta(p,3,1),0,0.5)$value
```

Out:

```
0.125
```

---

**Example**: _Comparing reviews_

Suppose one reseller has 80 positive reviews out of 100. The other reseller has two reviews, both positive. You could say the one with 100% approval is better?

We can express pdf using Gamma function: 


$f(x) = \frac{\Gamma(a+b)}{\Gamma(a) \Gamma(b)} x^{a-1}(1-x)^{b-1}$

**First case**: 80 positive out of 100 is $\operatorname{Beta}(81,21)$

$\theta_1 \text{PDF}:\frac{101!}{80!20!} x^{80} (1-x)^{20}$

**Second case**: 2 positive out of 2 is $\operatorname{Beta}(3,1)$

$\theta_2 \text{PDF}:3y^{2}$

To find the $Pr[\theta_1 > \theta_2]$ we integrate over region $x>y$:

$\int_{0}^{1} \int_{0}^{x} \frac{3 x^{80}(1-x)^{20} y^{2}}{\mathrm{~B}(81,21)} d y d x=\frac{91881}{182104} \approx 0.504552$

It is still better to purchase from the reseller with 80 positive review.



## Gamma distribution

In mathematics, the gamma function is an extension of the factorial function to complex numbers:

$\operatorname{Gamma}(\gamma \mid a, b)=\Large \frac{b^{a}}{\Gamma(a)} \gamma^{a-1} e^{-b \gamma}$

Where the $\Gamma(a)$ function is defined as:
$$
\Gamma(x) = \int_{0}^{\infty} u^{x-1} e^{-u} d u
$$

$\mathbb{E}[X]=a / b$

$\mathbb{V}[X]=a / b^{2}$

$\operatorname{Mode}[X]=\frac{a-1}{b}$

> Distributions showing only a single peak are called unimodal, bimodal distributions show two peaks in their frequency diagrams.

Gamma distribution is conjugate to the normal with respect to the precision.

>Precision is inverse of the variance.

$\gamma=\frac{1}{\sigma^{2}}$


Here is the PDF of the normal distribution:

$\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-\frac{(x-\mu)^{2}}{2 \sigma^{2}}}$

If we replace variance with inverse precision we get:

$\mathcal{N}\left(x \mid \mu, \gamma^{-1}\right)=\frac{\sqrt{\gamma}}{\sqrt{2 \pi}} e^{-\gamma \frac{(x-\mu)^{2}}{2}}$





