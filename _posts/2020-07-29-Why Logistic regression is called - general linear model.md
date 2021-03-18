---
published: false
layout: post
title: Why Logistic Regression is still a linear model
permalink: /logistic-regression-is-linear
---
Short answer because of the logit function.

Long answer:

A classifier is linear if its decision boundary on the feature space is a linear function: positive and negative examples are separated by an **hyperplane**.

This is what a SVM does by definition without the use of the kernel trick.

Also logistic regression uses linear decision boundaries. Imagine you trained a logistic regression and obtained the coefficients $\beta_i$. You might want to classify a test record $\mathbf{x} =(x_1,\dots,x_k)$ if $P(\mathbf{x}) > 0.5$. Where the probability is obtained with your logistic regression by:
$$P(\mathbf{x}) = \frac{1}{1+e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_k x_k)}}$$
If you work out the math you see that $P(\mathbf{x}) > 0.5$ defines a hyperplane on the feature space which separates positive from negative examples.

With $k$NN you don't have an hyperplane in general. Imagine some dense region of positive points. The decision boundary to classify test instances around those points will look like a curve - not a hyperplane.



