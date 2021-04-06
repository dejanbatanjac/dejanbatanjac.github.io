---
published: false
layout: post
title: Bayesian prerequisites | Model
permalink: /bayesian-model
---

- [IID](#iid)
- [Probability math](#probability-math)
- [MLE](#mle)
- [MAP and how to avoid the evidence](#map-and-how-to-avoid-the-evidence)
- [MAP problems](#map-problems)
- [Overcome MAP problems  -- conjugate posterior](#overcome-map-problems-----conjugate-posterior)
- [Conjugate prior for a likelihood function](#conjugate-prior-for-a-likelihood-function)
- [Beta prior, binomial likelihood](#beta-prior-binomial-likelihood)
- [Beta prior bernoulli likelihood](#beta-prior-bernoulli-likelihood)
- [Beta prior, geometric likelihood](#beta-prior-geometric-likelihood)
- [Normal prior normal likelihood](#normal-prior-normal-likelihood)






## IID

If observations $x^{(1)}, \ldots, x^{(N)}$  of a random variable $X$ are **independent and identically distributed** (i.i.d.) there is no dependence between the observations. This can be expressed as:


$$
p\left(x^{(1)}, \ldots, x^{(N)} \mid \theta\right)=\prod_{n=1}^{N} p\left(x^{(n)} \mid \theta\right)
$$


Here is how Wikipedia express it: 

In probability theory and statistics, a collection of random variables is independent and identically distributed if each random variable has the same probability distribution as the others and all are mutually independent.

_In other words_ for more than two random variables:

* **Independent**  refers to mutually independent random variables, where all subsets are independent. 

* **Identically distributed** means that all the random variables are from the same distribution.



## Probability math

The next rule expresses how the joint probability equals the conditional probability.

$$P(a,b)=P(a|b)P(b)$$

From there we can in fact derive the Bayes rule.


To the previous equation we can add another condition $\mid c$

$$P(a,b \mid c)=P(a|b,c)P(b \mid c)$$

We can prove the upper equation if we express both sides as joint probability.

$$\frac{P(a,b,c)}{P(c)} = \frac{P(a,b,c) P(b,c)}{P(b,c)P(c)}$$ 


## MLE

Maximum likelihood solution is equivalent to setting $\theta$ to that value that maximizes the likelihood of observing the data:

$$
\begin{aligned} \theta_{MLE} &=\arg \max_{\theta} P(X \mid \theta)
\end{aligned}
$$



MLE is summarization of the likelihood assuming a flat prior $p(\theta)= \mathsf{const}.$

## MAP and how to avoid the evidence

Maximum a posteriori is a summarization of the posterior:


$$
\begin{aligned} \theta_{MAP} &=\arg \max_{\theta} P(\theta \mid X)
\end{aligned}
$$

This is why the name.

Sometimes it is hard to compute the upper formula so we switch to :

$$\begin{aligned}\theta_{MAP} &=\arg \max _{\theta} \frac{P(X \mid \theta) P(\theta)}{P(X)}\end{aligned}$$

Which is equivalent to:

$$\begin{aligned}\theta_{MAP} &=\arg \max _{\theta} P(X \mid \theta) P(\theta) \end{aligned}$$

And for this problem we don't need the evidence. 

> For a **flat prior** $p(\theta)$ MAP is equivalent to MLE.



## MAP problems

We tried to avoid computing the evidence to get $\theta$, but there are problems.

Problems:

* If we apply nonlinear function to the input random variable $X \sim \mathcal N$, position of $\theta_{MAP}$ will change comparing to before the nonlinear function.

This means we cannot solve MAP problems except in certain cases where nonlinear function is Gaussian.


* We cannot use the posterior as the input again, so **no online learning**.
* Once we predict the $\theta_{MAP}$ we are not sure **how confident** we are
* position for $\theta_{MAP}$ may not be the indicator of the likelihood distribution. It is frequently **atypical** point (not in the center of the probability density).

> Note likelihood integral may not add to 1, it's not a probability.

## Overcome MAP problems  -- conjugate posterior

To overcome the problems with MAP we can *pair prior and the likelihood* to get the **conjugate posterior**.

Advantages would be:

* exact posterior calculation
* easy on-line learning

Constrains
* Conjugate prior may be inadequate for the problem we are solving.




## Conjugate prior for a likelihood function

Prior is said to be **conjugate for a likelihood** function if the posterior would stay in the same family of distributions as prior.

Why conjugate priors are useful? 

Because there is _no need for computing integrals_ because we get posterior in explicit form.

Finding a conjugate prior is useful because:

If posterior stays in the same family with prior, the integral $p\left(x_{n e w} \mid x\right)=\int p\left(x_{n e w} \mid \theta\right) p(\theta \mid x) d \theta$ for prediction is also tractable.

This integral is called **the evidence** and it can be computed analytically if prior, likelihood and posterior are known.

> We can perform analytical inference and find posterior distribution instead of taking point MAP estimate.


Conjugate matching

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



**Example:** What does it mean to know the distribution?

Imagine that you are training a neural network to play games. X is an image of the game screen and $\theta$ are parameters. 

If we know $P(X)$, we could generate new game-like frames.


