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



