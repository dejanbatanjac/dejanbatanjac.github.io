---
published: true
layout: post
title: Bayesian prerequisites | Bayes Rule
permalink: /bayesian-rule
---

## Bayes Rule

Bayes rule is given like this:

$$p(\theta \mid X) =\frac{p(X \mid \theta) p(\theta)}{p(X)}$$

Where: 
* $\theta$ are parameters
* $X$ observations
* $p(X)$ the evidence
* $p(X \mid \theta)$ likelihood (how well parameters explain our data)
* $p(\theta)$ Prior
* $p(\theta \mid X)$ posterior (probability of parameters after we observer the data)

In here we use small $p$ for probability of continuous distributions and $P$ for discrete distributions.

**Probability** tells us what is the chance of something given a data distribution.


## Probabilistic model

In here we introduce the concept of probabilistic model together with the concepts of:
* the likelihood 
* prior 
* posterior
* MLE (Maximum Likelihood Estimation) and 
* MAP (Maximum A Priori Estimation)

**Probabilistic model** is joint distribution of all its random variables.


**Likelihood** tells us given some parameter what is the best distribution of data. Likelihood is similar to the concept of loss function in classical approach.

**Prior** is similar to the concept of regularization in classical appraoch

**Posterior** is more specific knowledge of parameters after we selected prior knowledge that we get using the Bayes rule. The process to get the posterior from prior we call inference which is a cute name for learning.

**Evidence** also called marginal likelihood is what you get when you inegrate

$$p(X)=\int_\theta p(X \mid \theta) \,p(\theta)\, d\theta$$

Similar to posterior on **model parameters** we can compute posterior on **latent variables**, but then we need prior on latent variables $p(z)$.


$$p(z \mid X) =\frac{p(X \mid z) p(z)}{p(X)}$$

The problem on inference with latent variables is that we need likelihood on latent variables.

$$p(X \mid z) = \int_\theta p(X \mid z,\theta)p(\theta)d(\theta)$$ 

Sometimes is easier to compute **posterior distribution on latent variables** conditioned on model parameters:

$$p(z \mid X, \theta) =\frac{p(X \mid z, \theta) p(z)}{p(X \mid \theta)}$$


To marginalize out latent variables:

$$p(X \mid \theta) = \int_z p(X \mid z,\theta)p(z)d(z)$$ 


MLE and MAP are parameter estimators and the process to find parameters is called **Bayesian optimization**.

**MLE** is a function seeking for a model as a set of parameters that best fits the data. MLE uses likelihood function (minimum of the negative log likelihood function) to estimate the fit.

**MAP** is using minimum of the negative log posterior function to estimate Maximum a Posteriori fit. We say also MAP is summarization of the posterior.

Together with estimating parameters probabilistic models are used for **inference** and **prediction**. Later problems are connected with the difficult integrations unless we have **conjugate prior and posterior**.

Probabilistic model is specified by the joint distribution of all its random variables. This joint distribution will not show how random variables are connected together. For RV inner connection we use graphs.


## Conjugate prior for a likelihood function

Prior is said to be **conjugate for a likelihood** function if the posterior would stay in the same family of distributions as prior.



Matches:

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


