---
published: false
layout: post
title: Bayesian prerequisites | Full Bayesian Inference
permalink: /bayesian-full-inference
---


### Full Bayesian inference

Full Bayesian inference means we threat both the parameters and latent variables as **latent variables**.

In other words, everything is latent variables.

For example $x$ may be some image and $y$ may be category: _cat_ or _dog_, and $w$ are weights of neural net, threated as latent variables.

$$
\begin{aligned}
p(y \mid x, Y_{\text {train }}, X_{\text {train }}) 
&=\int p(y \mid x, w) p(w \mid Y_{\text {train }}, X_{\text {train }}) dw  \\
&=\mathbb{E}_{p(w \mid Y_{\text {train }}, X_{\text {train }})} p(y \mid x, w) 
\end{aligned}
$$

In here $p(y \mid x, w)$ is the neural network that takes inputs $x$ and outputs labels $y$.

Now the trick:
Instead of using just single neural network with set of weights $w$ we use big assemble of neural nets (say 100 different nets) where the weights are given as posterior distribution $p(w \mid Y_{\text {train }}, X_{\text {train }}) dw$.

This is called full Bayesian inference where instead fixed weights on single model we have the distribution of weights on multiple models.

We cannot do analytical computation in here because we cannot track what is inside neural nets so we can say the output of this integral is expected value of neural network.

The next line shows how we can find the posterior distribution of $w$ given the data set up to the normalization constant $Z$.

$$
p\left(w \mid Y_{\text {train }}, X_{\text {train }}\right)=\frac{p\left(Y_{\text {train }} \mid X_{\text {train }}, w\right) p(w)}{Z}
$$

On the right side everything is known:
* the prior $p(w)$ is usually the Gaussian 
* the likelihood part is the output of the neural net
