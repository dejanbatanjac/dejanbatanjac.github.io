---
published: false
layout: post
title: Bayesian Network
permalink: /probmodel
---

In here probabilistic model called **Bayesian Network** will be examined.

## What is Bayesian network?

It is a graph where random variables are nodes and edges represent the parent-child relationship or **impact** and **circular dependencies** are _prohibited_.

![ice cream2](/images/2021/03/ice2.png)

Or simplified:

![ice cream3](/images/2021/03/ice3.png)

We can write parent relationships:


$Pa(I) = \{R, S\} \\
Pa(S) = \{R\}$

The probability model for the upper image will be:

$\mathbb P(H,R,I) = \mathbb P(H)\mathbb P(R \mid H) \mathbb P(I \mid H, R)$


Easy to remember is the: Grandfather, Father, Son relation.

![gps](/images/2021/03/gps.png)

If we would write the model:

$\mathbb P(G,F,S) = \mathbb P(S)\mathbb P(F \mid G) \mathbb P(S \mid G, F)$


In general case joint probability model will be:

$$\begin{aligned}\mathbb P(X_{1}, \ldots, X_{n})=\prod_{k=1}^{n} \mathbb P(X_{k} \mid {Pa}(X_{k}))\end{aligned}$$ 

where $n$ is the number of nodes.

## 

![ice cream3](/images/2021/03/icen.png)

It is often in use to write plate notation for the upper image;

![ice cream3](/images/2021/03/icen_plate.png)

