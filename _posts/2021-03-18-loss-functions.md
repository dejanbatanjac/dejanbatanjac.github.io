---
published: false
layout: post
title: Loss functions 
permalink: /loss-functions
---

In here I will explain all the loss functions. Happens to be that I use PyTorch and I will mark the corresponding PyTorch function names.

## nn.MSELoss()

Criterion that measures the **mean squared error** also know as $\mathrm{L2}$ norm.

The unreduced (i.e. with reduction set to 'none ') loss can be described as:
$$
\ell(y, \hat y)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{\top}, \quad l_{n}=\left(y_{n}-\hat y_{n}\right)^{2}
$$
where $N$ is the batch size.

## L1Loss

This loss is for noisy data. 

## SmoothL1Loss 

Also know the Huber loss. Like L1Loss it is there to protect from outliers, but important replacement is in support $[-1,1]$ where MSE loss is used. The reason to do this is because MSE loss is differentiable for $x=1$.

$\operatorname{loss}(y, \hat y)=\frac{1}{n} \sum_{i} z_{i}$

Where: $z_{i}=\left\{\begin{array}{ll}0.5\left(y_{i}-\hat y_{i}\right)^{2}, & \text { if }\left|y_{i}-\hat y_{i}\right|<1 \\ \left|y_{i}-\hat y_{i}\right|-0.5, & \text { otherwise }\end{array}\right.$


## BCELoss

## BCEWithLogitsLoss

It is just the BCELoss but with log softmax in front.


## poisonNNLLoss()

## TripletMarginLoss

It is used for siamese net training.

## SoftMarginLoss

## nn.MultiLabelMarginLoss()

This is again the ranking loss but instead of insisting on just one correct category you can have multiple correct categories.

In here you can set the number of categories where you wish high score.

It is hinge loss in here but we do sum of ... over the whole categories and for each category if a categori is desired we push up, else down.

## nn.MultiLabelSoftMarginLoss()

## nn.MultiMarginLoss()


## nn.HingeEmbeddingLoss()

This is a loss for siamese nets, that pushes 

$y$ variable tells us if we should push up or down


## nn.CosineEmbeddingLoss()

This loss is normalized Euclidean distance.
Siamese nets.


## nn.CTCLoss()

