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
* $p(\theta \mid X)$ posterior (probability of parameters after we observe the data)
 
In here we use small $p$ for probability of continuous distributions and $P$ for discrete distributions.
 
**Probability** tells us what is the chance of something given a data distribution.
 
 
## Probabilistic model
 
In here we introduce the concept of probabilistic model together with the concepts of:
* the likelihood 
* prior 
* posterior
* MLE (Maximum Likelihood Estimation) and 
* MAP (Maximum A Priori Estimation)
 
A **probabilistic model** is fully specified by the joint distribution of all its random variables.
 
 
**Likelihood** is a function of the parameters (for fixed observed data). It quantifies how probable the observed data is for different values of the parameters. MLE consists of choosing the parameters that maximize this likelihood (or, equivalently, minimize the negative log-likelihood, which acts like a loss function).
 
**Prior** encodes our beliefs about the parameters before seeing the data. In MAP estimation the prior term acts as a regularizer (the negative log-prior penalizes certain parameter values).
 
**Posterior** represents our updated beliefs about the parameters after seeing the data. It is obtained by combining the prior with the likelihood via Bayes' rule. The process of obtaining the posterior (or computing quantities from it) is called **inference**.
 
**Evidence** (also called the marginal likelihood) is the probability of the observed data after integrating out the parameters (or latent variables). It is what you get when you compute
 
$$p(X)=\int_\theta p(X \mid \theta) \,p(\theta)\, d\theta$$
 
We can also write Bayes' rule for the posterior over latent variables $z$ (instead of parameters $\theta$):
 
 
$$p(z \mid X) =\frac{p(X \mid z) p(z)}{p(X)}$$
 
When latent variables are present, the marginal likelihood $p(X \mid z)$ itself usually requires an integral over the parameters (or other latents).
 
$$p(X \mid z) = \int_\theta p(X \mid z,\theta)p(\theta)d(\theta)$$ 
 
Sometimes it is easier to compute **posterior distribution on latent variables** conditioned on model parameters:
 
$$p(z \mid X, \theta) =\frac{p(X \mid z, \theta) p(z)}{p(X \mid \theta)}$$
 
 
To obtain the marginal likelihood w.r.t. the parameters (integrating out latents):
 
$$p(X \mid \theta) = \int_z p(X \mid z,\theta)p(z)d(z)$$ 
 
 
**MLE (Maximum Likelihood Estimation)** and **MAP (Maximum A Posteriori)** are two common methods for **point estimation** of model parameters.

- **MLE** finds the parameter values that make the observed data most probable (it maximizes the likelihood, or equivalently minimizes the negative log-likelihood).
- **MAP** does the same but also incorporates the prior; it maximizes the posterior probability. The prior term effectively acts as a regularizer.

**Important distinction**:
MLE and MAP are **parameter estimation** techniques. They give you a single "best" value for each parameter.

**Bayesian Optimization**, on the other hand, is a *black-box optimization* method. It is typically used to find good values for *hyperparameters* (learning rate, number of layers, regularization strength, etc.) when the objective function is expensive to evaluate. It usually relies on a surrogate model (e.g. Gaussian Process) + an acquisition function.

These are related ideas in the broader Bayesian world, but they are not the same thing.

**Quick comparison**:

| Method                  | Goal                                      | Finds model parameters? | Typically used for                  | Key tool                          |
|-------------------------|-------------------------------------------|---------------------------|-------------------------------------|-----------------------------------|
| **MLE**                 | Point estimate of parameters              | Yes                       | Classical parameter fitting         | Likelihood / negative log-likelihood |
| **MAP**                 | Point estimate of parameters (with prior) | Yes                       | Bayesian parameter estimation       | Posterior (prior acts as regularizer) |
| **Bayesian Optimization** | Optimize an expensive black-box function | No (optimizes *hyper*parameters) | Hyperparameter tuning (e.g. learning rate, architecture) | Gaussian Process + acquisition function |

MLE/MAP give you "best guess" values for the actual model parameters θ. Bayesian Optimization is an *optimization algorithm* you run on top of your training/validation procedure to choose good hyperparameter settings.

Together with point estimation (MLE / MAP), probabilistic models are used for **inference** (computing or approximating full posteriors over parameters or latent variables) and **prediction**. These tasks often involve intractable integrals. They become tractable when we have a **conjugate prior** to the likelihood.
 
A probabilistic model is fully specified by the joint distribution over all its random variables. However, the joint distribution alone does not explicitly reveal the conditional independence relationships between the variables. Graphical models (Bayesian networks or Markov networks) are used to visualize and exploit the dependency structure among the random variables (RVs).
 
 
## Conjugate prior for a likelihood function
 
A prior is said to be **conjugate** to a likelihood function if the resulting posterior belongs to the same distributional family as the prior. This makes many Bayesian calculations tractable in closed form.
 
 
 

 
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
 
 
 
 

