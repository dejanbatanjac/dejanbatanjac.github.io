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
- [What is Bayes method](#what-is-bayes-method)
- [Empirical Bayes methods](#empirical-bayes-methods)
- [Estimators. Biased and unbiased estimators](#estimators-biased-and-unbiased-estimators)
- [How to calculate $P(\mathcal D ; \theta )$](#how-to-calculate-pmathcal-d--theta-)
- [How unsupervised learning is different than supervised](#how-unsupervised-learning-is-different-than-supervised)
- [Estimating RV](#estimating-rv)
- [Comparing LR and NP](#comparing-lr-and-np)
- [Comparing LR and SVM](#comparing-lr-and-svm)

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

## What is Bayes method

Bayes method used Bayes rule for inference.


## Empirical Bayes methods

The prior distribution is fixed, as we said before for **Standard Bayes methods**, and if the prior distribution is not fixed but estimated from the data instead we have **Empirical Bayes methods**.


## Estimators. Biased and unbiased estimators

An estimator is an approximation of some parameter of the distribution. It can be either biased or unbiased.

If an estimator provides overestimate or underestimate of the true parameter it is called biased estimator.

If an estimator is accurate and no overestimate or underestimate exists it is called unbiased estimator.


## How to calculate $P(\mathcal D ; \theta )$
 
In practice it is hard to solve
 
$P(\mathcal D ; \theta)$
 
but if we assume:
 
$P(\mathcal D ; \theta) = \prod_{i} P(\mathcal D_{i} \mid \theta)$, where $\mathcal D_{i}$ is a particular value from a dataset.
 
We get a simple formula to calculate the probability. In many cases this works. Why?
 
Let's take an example evaluating if an email is ham or spam using NB. We can get good results even if we assume the order of words is not important. -->
 
<!-- ## Again : Estimate distribution from data
 
How do we estimate distribution P from data, because if we know to do that we can do magical things. 
 
There are several ways that are super pragmatic: 
 
 * MLA (Maximum Likelihood Estimation)
 * MAP (Maximum A Posteriori Estimation)
 
First we assume data has some form or distribution. This is why it is important to know different distributions and to understand the parameters that describe them. 
 
> Some of the distributions we often use are Binomial, Multinomial, Gaussian (and other exponential distributions), Gamma, etc. All distributions can really be either continuous or discrete.
 
 
 
### For MLA: $P(\mathcal D;\theta)$
 
We read this probability of data, parametrized by 
parameters $\theta$.
 
This approach is a frequentist approach. $\theta$ in here is a set of parameters.
 
### For MAP: $P(\boldsymbol{\theta} | \mathcal D)$
 
We read this as the probability of the $\boldsymbol{\theta}$ given dataset $\mathcal D$.
$\boldsymbol{\theta}$ is promoted here to a random variable and if we use Bayes rule we get:
 
 
$P(\boldsymbol{\theta} | \mathcal D) = \large \frac{P(\mathcal D | \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(\mathcal D)}$, but really since ${P(\mathcal D)}$ is a constant
 
> Extra note that MAP contains what is in MLA.
 
$P(\boldsymbol{\theta} | \mathcal D) \propto P(\mathcal D | \boldsymbol{\theta})P(\boldsymbol{\theta})$
 
We use the term likelihood as a synonym to probability; the only difference likelihood is used for things that already happened. 
 
MAP searches for the most likely model or set of $\theta$ that is most likely.
 
The MAP formula we used above is just a generalization of the truly Bayesian approach, that integrate over all possible models with the parameters $\boldsymbol \theta$:
 
 
$P(Y=y|X=x) = \int_{\boldsymbol{\theta}}P(Y=y|X,\boldsymbol{\theta})P(\boldsymbol{\theta} | \mathcal D)d\boldsymbol \theta$
 
In the majority of cases it is hard to get this integral and what we do instead is to sample 10.000 of different $\theta$ to get the average sum.
  -->
 
<!-- ## Classification (probabilistic definition)
 
* $P(y|\mathbf x,\mathcal{D}, M)$ - probability distribution over possible labels
 
* $M$ - the model, 
* $\mathcal{D}$ - the dataset
* $\mathbf x$ - vector of features
* $y$ -label 
 
 
$\hat{y}=\hat{f}(\mathbf{x})=\overset {C}{\underset{c=1}{\operatorname{argmax}}} P(y=c \mid \mathbf{x}, \mathcal{D}, M)$
 
If we have classification problem and just two classes $y=1$ and $y=0$ it is sufficient to define just:
 
$p(y=1 \mid \mathbf x, \mathcal D, \mathcal M)$ since we know the sum of probabilities is 1.
 
If $P(\hat{y} \mid \mathbf{x}, \mathcal{D}, M)=1$ we are confident about the answer. $\hat y$ is our true class.
 
## How unsupervised learning is different than supervised
 
First, we have written $P(\mathbf x_i \mid \theta)$ instead of $P(y_i|\mathbf x_i,θ)$ that is,
 
supervised learning is conditional density estimation, where unsupervised learning is unconditional density estimation.
 
 
Second, $\mathbf x_i$ is a vector of features, so we need to create _multivariate probability models_. 
 
In contrast, in supervised learning, $y_i$ is usually just a single variable which uses _univariate probability models_.
 
## Estimating RV
 
Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) estimation, are methods for random variable estimation.
 
These are approximation methods from the true distribution $\sim P(\mathrm x, \mathrm y)$ using the parameter $\theta$.
 
Approximate: $D \sim P_{\theta}(X, Y)  = P(X,Y;\theta)$
 
Where $X$ and $Y$ are random variables.
 
 
Frequentists approach:
 
MLE: $\theta = \underset{\theta}{\arg \max} P(D;\theta)$
 
Bayesian approach:
 
MAP: $\theta = \underset{\theta}{\arg \max} P(\theta|D)$
 
 
No machine learning model.
 
$P(Y, X=x)=\int_{\theta}P(Y|\theta)P(\theta|D)d\theta$
 
 
$P(Y=y, X=x)$ is **hard** to compute. This is why we start using the Naive Bayes approach.
 
The assumption of Naive Bayes is that all features $X$ are mutually independent.
 
$P(Y=y| X=x) = \large \frac{P(X=x| Y=y)P(Y=y)}{P(X=x)}$
 
$P(Y=y)$ is usually easy to compute. $P(X=x)$ we may assume a constant.
 
So we have the solution now: $P(X=x, Y=y)$.
 
Because features are independent we may write:
 
$P(X=x| Y=y)=\prod_{\alpha=1}^d(X_{\alpha}=x_{\alpha} | Y=y)$
 
 
Now, we can create Bayes estimator (classifier)
 
$h(X) = \underset{Y}{arg max}P(Y|X) \\ 
= \underset{Y}{arg max} \large \frac{P(X|Y)P(Y)}{z} \\
= \underset{Y}{arg max} P(Y) \prod_{\alpha=1}^d P(X_{\alpha}|Y) \\
= \underset{Y}{arg max} \log P(Y) + \sum_{\alpha=1}^d \log P(X_{\alpha}|Y) \\
$
 
The last sum is easy to estimate since it is based on 1d.
 
## Comparing LR and NP
 
LR separates the dataset points. NP separates the data distribution that fits the data.
 
With a low number of data we may expect NP may work better. If we have a large amount of data it may be better for the data to speak for itself, so LR would be a better fit.
 
>We can always check if the distribution assumption we choose is good or bad. Once we select the distribution, Gaussian, Binomial, Multinomial, ... we may create the test points and if these points confirm they are a good match with the original data we have the right to say the distribution is well chosen. 
 
## Comparing LR and SVM
 
LR can also provide the probability estimation, not just the class, while SVM cannot provide the probability estimation. 
 
For instance, SVM is not a good match for self-driving cars. SVM will tell us if something is pedestrian or not, but it cannot tell us the probability of how likely it is the case. For instance it may be 49% chance it is a pedestrian, and 51% it is not. SVM will just tell us it is not a pedestrian.
 
One of the great features with LR is it can actually tell us the estimated probabilities. At some point it is possible to combine the SVM and LR.
 
SVM can provide us the single feature (which class we predict) and LR will take this feature and predict the probability. In fact this system is known as Platt scaling, named by John Platt.

 