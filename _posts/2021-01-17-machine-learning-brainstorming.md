---
published: false
layout: post
title: ML brainstorming 
permalink: /ml-brainstorming
---
- [Types of machine learning](#types-of-machine-learning)
- [Why do we have train, validation and test set](#why-do-we-have-train-validation-and-test-set)
- [What is overfitting](#what-is-overfitting)
- [What is the cross validation](#what-is-the-cross-validation)
- [Problem with cross validation when the number of features is big and number of samples is small](#problem-with-cross-validation-when-the-number-of-features-is-big-and-number-of-samples-is-small)
- [Parametric vs. non parametric models](#parametric-vs-non-parametric-models)
- [Generative vs. discriminative learning](#generative-vs-discriminative-learning)
- [Probability vs. likelihood what is the difference](#probability-vs-likelihood-what-is-the-difference)
- [The two approaches to probability](#the-two-approaches-to-probability)
- [What is Bayesian approach](#what-is-bayesian-approach)
- [What is random variable](#what-is-random-variable)
- [How you estimate the distribution from data](#how-you-estimate-the-distribution-from-data)
- [What is the difference MLE and MAP](#what-is-the-difference-mle-and-map)
- [How to calculate $P(\mathcal D ; \theta )$](#how-to-calculate-pmathcal-d--theta-)
- [Again : Estimate distribution from data](#again--estimate-distribution-from-data)
  - [For MLA: $P(\mathcal D;\theta)$](#for-mla-pmathcal-dtheta)
  - [For MAP: $P(\boldsymbol{\theta} | \mathcal D)$](#for-map-pboldsymboltheta--mathcal-d)
- [Classification (probabilistic definition)](#classification-probabilistic-definition)
- [How unsupervised learning is different than supervised](#how-unsupervised-learning-is-different-than-supervised)
- [Estimating RV](#estimating-rv)
- [Comparing LR and NP](#comparing-lr-and-np)
- [Comparing LR and SVM](#comparing-lr-and-svm)
- [Gradient descent vs. Newton method](#gradient-descent-vs-newton-method)
- [GD vs Adagrad](#gd-vs-adagrad)

This will be the Machine learning brainstorming
(probabilistic approach).
## Types of machine learning

Machine learning is usually divided into two main types:

* predictive or supervised learning
* descriptive or unsupervised learning


Predictive learning has the mapping from input $\mathbf x$ to output $y$.

$\mathcal{D}=\left\{\left(\mathbf{x}_{i}, y_{i}\right)\right\}_{i=1}^{N}$

Where $N$ is the number of training examples, $y_i$ is the single output feature, $\mathbf{x}_{i}$ is all the input features and $\mathcal{D}$ is the training set.

Descriptive learning doesn't have the target.

$\mathcal{D}=\left\{\mathbf{x}_{i}\right\}_{i=1}^{N}$

It is sometimes called _knowledge discovery_ and our goal is to find _data patterns_.

The problem with descriptive learning is there is no obvious error metric to use.

> There is a third type of machine learning, known as _reinforcement_ learning 

## Why do we have train, validation and test set

While just train and validation sets may be enough, why do we also need the test set. Isn't it the case the validation set is enough.

We need all three sets, train set to train against it and validation set to check if we are good. 

But there need to be some never used data (read: test set) for our final check. If we would use the test set before, our machine learning parameters would be biased with the test data. 

The test set purpose is to reflect some real data, that we haven't seen before.


## What is overfitting

If you would train and test on the same data you do a _methodological mistake_.

The model that fails to predict well on unseen data probable has overfitting problem.

Overfitting is when the model learns the train data super well, and on unseen data it provides bad results.

## What is the cross validation

CV in the most basic form k-fold cross validation means the training set is split into k smaller sets.

In case of k-fold CV:

* the model is trained on `k-1` folds as training data
* the resulting model is validated on the remaining part of the data 


## Problem with cross validation when the number of features is big and number of samples is small 

There is one rule for cross validation if we have great number of features and low number of samples

Say we have 5000 features and 50 samples and we need to predict one of the two possible classes.

1. We could find top top 100 features (correlation) 

2. We could then use CV on those 100 features and 50 samples. 

But this would be methodological mistake again. 
We should do CV in step 1.


## Parametric vs. non parametric models

There are two types of models:
* parametric
* non parametric

If _the number of model parameters grows with the number of data_ this model is non parametric else parametric model.

Some parametric models:
* The Perceptron
* LR Linear Regression
* NB Naive Bayes
* SVM Support Vector Machines

The example of non parametric machine learning models:
* KNN 
* Decision Trees


## Generative vs. discriminative learning

_Discriminative algorithm_ given all measurement of a human will give you if the human is male of female.

_Generative algorithm_ given a gender will generate a human.

Discriminative model focuses on what distinguishes the two or more classes.

A generative model models the distribution (detailed characteristics of each class). 

SVM is an example of discriminative model.
NB is an example of generative model.


## Probability vs. likelihood what is the difference

Usually these are the same, but you cannot say what is the probability of the dataset because dataset already exists.

For something already present it is more likely that you will use the term _likelihood_.

Use _probability_ for something that hasn't happened yet.

## The two approaches to probability

The basic fact about the probability it cannot be greater than 1 and lower than 0. For mathematicians the probability objective is to express how certain is an event or statement.

The so called _frequentists approach_ is to measure all possible outcomes of an event and to derive some conclusions about the event. For instance, rolling a regular (unbiased) or fair dice. The estimated probability to get 1 is $\tfrac{1}{6}$ that we got after 1 million measures.

The other approach is know as _Bayesian probability approach_. For the same act rolling a fair dice, we may add some prior beliefs. We basically add our belief that we got 100 times 1, and then we start the measurements.

Else, if we don't have any prior beliefs at all, this is called the _uninformed prior_.

Bayesian approach is especially important when we cannot do many measurements, or when the measurements are costly. For example, to answer the questions how likely a person will catch a cold. 

## What is Bayesian approach

We often deal with uncertainty when we search for:

* the best prediction given data
* the best model given data
* what measurement should we perform next

The systematic application of probabilistic reasoning is called probabilistic approach or sometimes _Bayesian approach_.


## What is random variable

As it turns out the idea of random variable (also called stochastic) is very important when dealing with probabilities. Random variable can be discrete and continuous.

A finite collection of random variables defined under certain conditions is a random vector.

An infinite collection of random variables defined under certain conditions is a random process.

A latent variable is a random variable that we cannot observe directly.

A proper formal understanding of random variables is defined via a branch of mathematics known as measure theory. Measure theory defines terms such as:

* almost everywhere
* measure zero set

In short: A random variable is a variable whose possible values have an associated _probability distribution_.

Check more on [probability distributions](https://programming-review.com/math/probability-distributions)

## How you estimate the distribution from data

This is one of the main questions in machine learning (_probabilistic approach_)

There are few ways to estimate distributions from data:

* MLE (Maximum Likelihood Estimation)
* MAP (Maximum A Posteriori Estimation)
* Bayesian inference

For MLE if we have to maximize $P(\mathcal D \mid \theta)$, where $\theta$ is a set of parameters, 

For MAP we have $P(\theta \mid \mathcal D)$, where $\theta$ is random variable now.

$\theta$ need to be random variable so we can use the Bayes rule:

$P(\theta \mid \mathcal D) = \large \frac{P(\mathcal D \mid \theta )P(\theta)}{z}$, where $z$ is normalization constant.

In both cases we end with the pattern $P(\mathcal D \mid \theta )$ to maximize which means MLE and MAP will have similar algorithmic steps.

Bayesian inference returns probability density function and is complex to calculate.

$P(y_{t} \mid \mathbf x_{t}) = \int_{\theta} P(y_{t} \mid \mathbf x_{t} , \theta) P(\theta \mid \mathcal D) d\theta$ 

To calculate this integral is hard and this is why in practice we use pragmatic MLE and MAP approaches.

## What is the difference MLE and MAP

MAP is pragmatic approach to Bayesian optimization where we take say thousand different $\theta$s and we average those.

It's like systematically trying different parameters and defining what are the best parameters. If we have just two parameters imagine creating a grid with different values for all the parameters and finding the best parameters.

MLE gives the value which maximizes the likelihood $P(\mathcal D \mid \theta)$. And MAP gives you the value which maximizes the posterior probability $P(\theta \mid \mathcal D)$.

## How to calculate $P(\mathcal D ; \theta )$

In practice it is hard to solve

$P(\mathcal D ; \theta)$

but if we assume:

$P(\mathcal D ; \theta) = \prod_{i} P(\mathcal D_{i} \mid \theta)$, where $\mathcal D_{i}$ is a particular value from a dataset.

We get a simple formula to calculate the probability. In many cases this works. Why?

Let's take an example evaluating if an email is ham or spam using NB. We can get good results even if we take the assumption the order of words is not important.

## Again : Estimate distribution from data

How do we estimate distribution P from data, because if we know to do that we can do magical things. 

There are several ways that are super pragmatic: 
 
 * MLA (Maximum Likelihood Estimation)
 * MAP (Maximum A Posteriori Estimation)
 
First we assume data has some form or distribution. This is why it is important to know different distributions and to understand the parameters that describe them. 
 
> Some of the distribution we often use are Binomial, Multinomial, Gaussian (and other exponential distributions), Gamma, etc. All distributions can really be either continuous or discrete.

### For MLA: $P(\mathcal D;\theta)$

We read this probability of data, parametrized by 
parameters $\theta$.

This approach is frequentists approach. $\theta$ in here is a set of parameters.

### For MAP: $P(\boldsymbol{\theta} | \mathcal D)$

We read this what is probability of the $\boldsymbol{\theta}$ given dataset $\mathcal D$.
$\boldsymbol{\theta}$ is promoted here to a random variable and if we use Bayes rule we get:


$P(\boldsymbol{\theta} | \mathcal D) = \large \frac{P(\mathcal D | \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(\mathcal D)}$, but really since ${P(\mathcal D)}$ is a constant

> Extra note that MAP contains what is in MLA.

$P(\boldsymbol{\theta} | \mathcal D) \propto P(\mathcal D | \boldsymbol{\theta})P(\boldsymbol{\theta})$

We use the term likelyhood as a synonym to probability; the only difference likelyhood is used for things that already happened. 

MAP searches for the most likely model or set of $\theta$ that is most likely.

The MAP formula we used above is just a generalization of the truly Bayesian approach, that integrate over all possible models with the parameters $\boldsymbol \theta$:


$P(Y=y|X=x) = \int_{\boldsymbol{\theta}}P(Y=y|X,\boldsymbol{\theta})P(\boldsymbol{\theta} | \mathcal D)d\boldsymbol \theta$

In majority of cases it is hard to get this integral and what we do instead is to sample 10.000 of different $\theta$ to get the average sum.


## Classification (probabilistic definition)

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


$P(Y=y, X=x)$ is **hard** to compute. This is why we start using Naive Bayes approach.

The assumption of Naive Bayes is that all features $X$ are mutually independent.

$P(Y=y| X=x) = \large \frac{P(X=x| Y=y)P(Y=y)}{P(X=x)}$
 
$P(Y=y)$ is usually easy to compute. $P(X=x)$ we may assume a constant.

So we have the solve now: $P(X=x, Y=y)$.

Because features are independend we may write:

$P(X=x| Y=y)=\prod_{\alpha=1}^d(X_{\alpha}=x_{\alpha} | Y=y)$


Now, we can create Bayes estimator (classifier)

$h(X) = \underset{Y}{arg max}P(Y|X) \\ 
= \underset{Y}{arg max} \large \frac{P(X|Y)P(Y)}{z} \\
= \underset{Y}{arg max} P(Y) \prod_{\alpha=1}^d P(X_{\alpha}|Y) \\
= \underset{Y}{arg max} \log P(Y) + \sum_{\alpha=1}^d \log P(X_{\alpha}|Y) \\
$

The last sum is easy to estimate since it is based on 1d.

## Comparing LR and NP

LR separate the dataset points. NP separates the data distribution that fits to the data.

With low number of data we may expect NP may work better. If we have large number of data it may be better for the data to speak for itself, so LR would be a better fit.

>We can always check if the distribution assumption we choose is good or bad. Once we select the distribution, Gaussian, Binomial, Multinomial, ... we may create the test points and if these points confirms they are a good mach with the original data we have the right to say the distribution is well chosen. 

## Comparing LR and SVM

LR can provide also the probability estimation, not just the class, while SVM cannot provide he probability estimation. 

For instance, SVM is not a good match for self driving cars. SVM will tell us if something is pedestrian or not, but it cannot tell us the probability how likely is the case. For instance it may be 49% chance it is a pedestrian, and 51% it is not. SVM will just tell us it is not a pedestrian.

One of the great features with LR is it can actually tell us the estimated probabilities. At some point it is possible to combine the SVM and LR.

SVM can provide us the single feature (which class we predict) and LR will take this feature and predict the probability. In fact this system is know as Platt scaling, named by John Platt.

## Gradient descent vs. Newton method

If we use Taylor expansion linear approximation we are in domain of gradient descent.

GD is iterative optimization algorithm for finding a local minimum of a differentiable function.

If we use second order expansion we are in domain of Newton method.

There are pro and cons for Newton method.

Pros of the Newton method:

* fast convergence (in just a few steps) to the local minima

Cons of the Newton method:

* the Newton method may not converge (no luck)
* if the dimensionality of the data is huge it is hard to compute the Hessian ($D^2$), and inverse of Hessian ($D^3$), Where $D$ is the dimensionality of the data. 

> Note the dataset may have large number of features $f$ and large number of data samples $n$, but the dimensionality of the data may just be $D=10$

> In case we use diagonal of a Hessian, we can approximate Newton method. This way it is easy to compute the Hessian even for large values of $D$.

## GD vs Adagrad

GD as single step size, Adagrad has step size for every single dimension.





