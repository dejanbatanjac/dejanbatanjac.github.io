---
published: false
layout: post
title: Bayesian Linear Regression
permalink: /bayesiaon-linear-regression
---
- [Two ways](#two-ways)
- [Parameters and the Data](#parameters-and-the-data)
- [When the number is important](#when-the-number-is-important)
- [How do we train](#how-do-we-train)
- [Classification](#classification)
- [Prior as the Regularizer](#prior-as-the-regularizer)
- [Online learning](#online-learning)


## Define the linear regression 

Our approach to linear regression uses the MSE algorithm: 

$\begin{aligned}L(w)=\sum_{i=1}^{N}\left(w^{T} x_{i}-y_{i}\right)^{2}=\left\|w^{T} X-y\right\|^{2}\end{aligned}$

Our problem is therefore the minimization problem of $L(w)$.


$\begin{aligned}\therefore \widehat{w}=\arg \min _{w} L(w)\end{aligned}$

![lr](/images/2021/02/lr1.png)

$P(w, y \mid X)=P(y \mid X, w) P(w)$

$P(y \mid w, X)=\mathcal{N}\left(y \mid w^{T} X, \sigma^{2} I\right)$

$P(w)=\mathcal{N}\left(w \mid 0, \gamma^{2} I\right)$