---
published: true
layout: post
title: Bias Variance Noise trade off 
permalink: /bias-variance-noise-trade-off
---
- [The machine learning models](#the-machine-learning-models)
- [The dataset model](#the-dataset-model)
- [The dataset model again](#the-dataset-model-again)
- [The expected label (target)](#the-expected-label-target)
- [Hypothesis  $h_D$](#hypothesis--h_d)
- [Expected test error (given $h_D$)](#expected-test-error-given-h_d)
- [Expected hypothesis (given $\mathcal A$):](#expected-hypothesis-given-mathcal-a)
- [Expected test error (given $\mathcal A$)](#expected-test-error-given-mathcal-a)
- [Decomposition of the expected test error](#decomposition-of-the-expected-test-error)
- [Final Formula](#final-formula)
  - [Variance:](#variance)
  - [Bias](#bias)
  - [Noise](#noise)
- [Model complexity trade off](#model-complexity-trade-off)
- [Intuition on bias and variance](#intuition-on-bias-and-variance)
  - [How to reduce high variance?](#how-to-reduce-high-variance)
  - [How to reduce the high bias?](#how-to-reduce-the-high-bias)

One of the most important formula in statistics and machine learning is **bias variance trade off**. It actually **generalize** to **Bias, Variance Noise trade off** here I will show how.

Engineering job is to find the equilibrium of these three, since intuitively if you make one of them smaller, the other usually get bigger.

The following analysis will track one regression problem and explain the decomposition of the expected test error from the **probabilistic approach**.

## The machine learning models

According to **No Free Lunch Theorem** every successful machine learning algorithm must make assumptions about the model. 

In other words there is **no single machine learning algorithm** that works for all machine learning problems.

We write particular algorithm $h$ is a **hypothesis** drawn from a hypothesis class $\mathcal H$.


## The dataset model

Let's have a multivariate random variable or random vector $X$ and a target random variable $Y$.

The joint distribution of $X$ and $Y$ is also a random variable $D \sim P(X,Y)$. 

When we draw concrete dataset $\mathcal D$ is drawn from **sample space** $P(X,Y)^n$ we get our dataset $\mathcal D$ with $n$ rows.

> The domain of random variable is called a sample space. This is a set of all possible outcomes, but we don't bother much with it now.


## The dataset model again

Let's have the same dataset: $\mathcal D = \{(\mathbb{x}_i, y_i)\}_{i=1}^n$, where:
* $y \in \mathbb{R}$ is our **target** or **label** 
* $\mathbb{x} \in \mathbb{R}^d$ is our features vector (read: a row in a dataset not including the target) composed from $d$ columns
* training dataset has $n$ inputs (rows)

>$\mathbb R^d$ is the $d$-dimensional feature space, generally it may not be $\mathbb R$, but in here this is our simplified assumption.

> We tackle a typical regression problem, but the idea can generalize.



> Note in here we are assuming there is just a single realization $\mathcal D$ of a random variable $D$ for simplicity, even though I can draw infinitely meany datasets $\mathcal D$ from $D$. This is why I will just write $D$ from now on.


## The expected label (target)

If our problem is **selling cars**, there may be two cars, with the exactly same equipment, but the target price may differ.

This is why we need **the expected target** value $\bar y(x)$ where $\mathbb x$ is a vector of features.

$\bar{y}(x)=\mathbb E_{y \mid \mathbb{x}}[Y]=\int_{y} y P(y \mid \mathbb{x}) \partial y$


## Hypothesis  $h_D$

Since we have **training dataset** $D$ we may start the learning process.

> In here $\mathcal D = D$, random variable $D$ is the only realization.

$h_{D} = \mathcal A(D)$

Now, $D$ is a random variable and $\mathcal A$ is the algorithm so $h_D$ is also a random variable.


The specific meaning of this random variable $h_D$ is that it represents **all the hypothesis** from all hypothesis classes.

Now, it would be great if we can say what is the best class $h_D$? This is why we will from now on use the MSE loss function (most common for regression problems).


## Expected test error (given $h_D$)

$\begin{aligned}\mathbb E_{(\mathbb{x}, y) \sim P}\left[\left(h_{D}(\mathbb{x})-y\right)^{2}\right]=\int_{x}\int_{y}\left(h_{D}(\mathbb{x})-y\right)^{2} \operatorname{P}(\mathbb{x}, y) \partial y \partial \mathbb{x}\end{aligned}$

This is for specific hypothesis function on a specific dataset.


## Expected hypothesis (given $\mathcal A$): 

Note there are different possible hypothesis on a dataset. We typically need the best one (with the minimal loss) but we can think of the expected hypothesis and expected test error as well.


$\bar{h}=\mathbb E_{D \sim P^{n}}\left[h_{D}\right]=\int_{D} h_{D} \operatorname{P}(D) \partial D$


## Expected test error (given $\mathcal A$)


$\mathbb E_{(\mathbb{x}, y) \sim P}\left[\left(h_{D}(\mathbb{x})-y\right)^{2}\right]=\int_{D} \int_{\mathbb{x}} \int_{y}\left(h_{D}(\mathbb{x})-y\right)^{2} \mathrm{P}(\mathbb{x}, y) \mathrm{P}(D) \partial \mathbb{x} \partial y \partial D$


## Decomposition of the expected test error

$$
\begin{aligned}
\mathbb E_{\mathbb{x}, y, D}\left[\left(h_{D}(\mathbb{x})-y\right)^{2}\right] &=\mathbb E_{\mathbb{x}, y, D}\left[\left[\left(h_{D}(\mathbb{x})-\bar{h}(\mathbb{x})\right)+(\bar{h}(\mathbb{x})-y)\right]^{2}\right] \\
&=\mathbb E_{\mathbb{x}, D}\left[\left(\bar{h}_{D}(\mathbb{x})-\bar{h}(\mathbb{x})\right)^{2}\right]+2 \mathbb E_{\mathbb{x}, y, D}\left[\left(h_{D}(\mathbb{x})-\bar{h}(\mathbb{x})\right)(\bar{h}(\mathbb{x})-y)\right]+\mathbb E_{\mathbb{x}, y}\left[(\bar{h}(\mathbb{x})-y)^{2}\right]
\end{aligned}
$$
The middle term of the above equation is 0 as we show below
$$
\begin{aligned}
\mathbb E_{\mathbb{x}, y, D}\left[\left(h_{D}(\mathbb{x})-\bar{h}(\mathbb{x})\right)(\bar{h}(\mathbb{x})-y)\right] &=\mathbb E_{\mathbb{x}, y}\left[\mathbb E_{D}\left[h_{D}(\mathbb{x})-\bar{h}(\mathbb{x})\right](\bar{h}(\mathbb{x})-y)\right] \\
&=\mathbb E_{\mathbb{x}, y}\left[\left(\mathbb E_{D}\left[h_{D}(\mathbb{x})\right]-\bar{h}(\mathbb{x})\right)(\bar{h}(\mathbb{x})-y)\right] \\
&=\mathbb E_{\mathbb{x}, y}[(\bar{h}(\mathbb{x})-\bar{h}(\mathbb{x}))(\bar{h}(\mathbb{x})-y)] \\
&=\mathbb E_{\mathbb{x}, y}[0] \\
&=0
\end{aligned}
$$


Returning to the earlier expression, we're left with the variance and another term:


$$
\mathbb E_{\mathbb{x}, y, D}\left[\left(h_{D}(\mathbb{x})-y\right)^{2}\right]=\underbrace{\mathbb E_{\mathbb{x}, D}\left[\left(h_{D}(\mathbb{x})-\bar{h}(\mathbb{x})\right)^{2}\right]}_{\text {Variance }}+\mathbb E_{\mathbb{x}, y}\left[(\bar{h}(\mathbb{x})-y)^{2}\right]
$$


We can break down the second term in the above equation as follows:


$$
\begin{aligned}
\mathbb E_{\mathbb{x}, y}\left[(\bar{h}(\mathbb{x})-y)^{2}\right] &=\mathbb E_{\mathbb{x}, y}\left[(\bar{h}(\mathbb{x})-\bar{y}(\mathbb{x}))+(\bar{y}(\mathbb{x})-y)^{2}\right] \\
&=\underbrace{\mathbb E_{\mathbb{x}, y}\left[(\bar{y}(\mathbb{x})-y)^{2}\right]}_{\text {Noise }}+\underbrace{\mathbb E_{\mathbb{x}}\left[(\bar{h}(\mathbb{x})-\bar{y}(\mathbb{x}))^{2}\right]}_{\text {Bias }^{2}}+2 \mathbb E_{\mathbb{x}, y}[(\bar{h}(\mathbb{x})-\bar{y}(\mathbb{x}))(\bar{y}(\mathbb{x})-y)]
\end{aligned}
$$

The third term in the equation above is $0,$ as we show below:

$$
\begin{aligned}
\mathbb E_{\mathbb{x}, y}[(\bar{h}(\mathbb{x})-\bar{y}(\mathbb{x}))(\bar{y}(\mathbb{x})-y)] &=\mathbb E_{\mathbb{x}}\left[\mathbb E_{y \mid \mathbb{x}}[\bar{y}(\mathbb{x})-y](\bar{h}(\mathbb{x})-\bar{y}(\mathbb{x}))\right] \\
&=\mathbb E_{\mathbb{x}}\left[\mathbb E_{y \mid \mathbb{x}}[\bar{y}(\mathbb{x})-y](\bar{h}(\mathbb{x})-\bar{y}(\mathbb{x}))\right] \\
&=\mathbb E_{\mathbb{x}}\left[\left(\bar{y}(\mathbb{x})-\mathbb E_{y \mid \mathbb{x}}[y]\right)(\bar{h}(\mathbb{x})-\bar{y}(\mathbb{x}))\right] \\
&=\mathbb E_{\mathbb{x}}[(\bar{y}(\mathbb{x})-\bar{y}(\mathbb{x}))(\bar{h}(\mathbb{x})-\bar{y}(\mathbb{x}))] \\
&=\mathbb E_{\mathbb{x}}[0] \\
&=0
\end{aligned}$$

## Final Formula

**Total error** can be decomposed to **bias**, **variance** and **noise**.
<div>

$\begin{aligned}\underbrace{\mathbb E_{\mathbb{x}, y, D}\left[\left(h_{D}(\mathbb{x})-y\right)^{2}\right]}_{\text {Expected Test Error }}= &\underbrace{\mathbb E_{\mathbb{x}, D}\left[\left(h_{D}(\mathbb{x})-\bar{h}(\mathbb{x})\right)^{2}\right]}_{\text {Variance }} \\ \\ &+\underbrace{\mathbb E_{\mathbb{x}, y}\left[(\bar{y}(\mathbb{x})-y)^{2}\right]}_{\text {Noise }}\\ \\ &+\underbrace{\mathbb E_{\mathbb{x}}\left[(\bar{h}(\mathbb{x})-\bar{y}(\mathbb{x}))^{2}\right]}_{\text {Bias }^{2}}\end{aligned}$
</div>

### Variance:

How much the model output changes if you train on a different dataset set $D$? 

This variance is the expectation of the squared difference from the hypothesis function on dataset $D$ and average hypothesis.

If we are **over-specialized** on some set $D$ we are **overfitting**.

Typically if the dataset becomes larger the variance should become smaller.

If I remove the outliers, I change the data distribution, and this can certainly help the variance decrease. This means I can use complex models, because variance decreased, and if I use complex models the bias will decrease. 

### Bias

Bias exists because your hypothesis functions 
are **biased** to a particular kind of solution.
In other words, bias is inherent to your model.

### Noise

How can we reduce noise? Noise depends on the number of features (feature space) so with more features can can help reducing the data inartistic noise.

Also noise is not a function of $D$, more data will not help reducing the noise.

However, here is the **trade off**, if I add more features, and make noise smaller, I will make the variance larger. This would be the **features trade off**.

## Model complexity trade off

![model complexity and bias variance tradeoff](/images/2021/02/biasvariance.v1.png)

If we set the noise aside the bias variance tradeoff, is simple this.

We know as our model is more complex, it is capable to learn the train dataset specifics well, but it will **overfit** on a test dataset. The variance on a test dataset will be big. Simpler models cannot learn all the fine grains of the training dataset and because of that the variance will be lower.

It is completely opposite from the bias perspective. The model complexity, make the bias lower. This provides the intuition there is the **optimal model complexity** we are looking for.


## Intuition on bias and variance

![bulls eye](/images/2021/02/bulls-eye.png)

### How to reduce high variance?

You can detect high variance if the training error is much lower than test error.

If this is the case:

* add more training data
* reduce model complexity
* use bagging

### How to reduce the high bias?

**If you are not in high variance mode you your train and test errors should be close.**

If you have high bias: the model being used is not robust enough to produce an accurate prediction.

If this is the case:

* use complex model
* add features
* use boosting


---
Reference: [1](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote12.html){:rel="nofollow"}, [2](http://scott.fortmann-roe.com/docs/BiasVariance.html){:rel="nofollow"}

---