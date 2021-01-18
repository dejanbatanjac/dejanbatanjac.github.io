---
published: true
layout: post
title: Probability notation
permalink: /probability-notation
---
- [Basic probability notation](#basic-probability-notation)
- [Conditional probability](#conditional-probability)
- [Chain rule of probabilities](#chain-rule-of-probabilities)
- [Notations of Random Variables](#notations-of-random-variables)

## Basic probability notation
The probability that $\mathrm x = x$ is denoted as $P(x)$

Sometimes we deﬁne a variable ﬁrst, then use $\sim$ notation to
specify which distribution it follows later: $\mathrm x ∼ P(\mathrm x)$

Probability mass functions can act on many variables at the same time known as **joint probability distribution**. 

$P(x, y) = P(\mathrm x=x, \mathrm y=y)$

For $u(x;a, b)$ we say $x$ is "parametrized by" $a$ and $b$.

## Conditional probability

$P(\mathrm{y}=y \mid \mathrm{x}=x)=\large \frac{P(\mathrm{y}=y, \mathrm{x}=x)}{P(\mathrm{x}=x)}$

This can be rewritten as:

$P(\mathrm{y} \mid \mathrm{x})=\large \frac{P(\mathrm{y}, \mathrm{x})}{P(\mathrm{x})}$

Or as:

$P(\mathrm{Y} \mid \mathrm{X})=\large \frac{P(\mathrm{Y}, \mathrm{X})}{P(\mathrm{X})}$


It means:

We are interested in the probability of event $\text Y$, given that some
other event $\text X$ has happened. This is called a conditional probability.


## Chain rule of probabilities

Any joint probability distribution over many random variables may be decomposed
into conditional distributions over only one variable:

$P\left(\mathrm{x}^{(1)}, \ldots, \mathrm{x}^{(n)}\right)=P\left(\mathrm{x}^{(1)}\right) \Pi_{i=2}^{n} P\left(\mathrm{x}^{(i)} \mid \mathrm{x}^{(1)}, \ldots, \mathrm{x}^{(i-1)}\right)$

_Example:_

$\begin{aligned} P(\mathrm{a}, \mathrm{b}, \mathrm{c}) &=P(\mathrm{a} \mid \mathrm{b}, \mathrm{c}) P(\mathrm{b}, \mathrm{c}) \\ P(\mathrm{b}, \mathrm{c}) &=P(\mathrm{b} \mid \mathrm{c}) P(\mathrm{c}) \\ P(\mathrm{a}, \mathrm{b}, \mathrm{c}) &=P(\mathrm{a} \mid \mathrm{b}, \mathrm{c}) P(\mathrm{b} \mid \mathrm{c}) P(\mathrm{c}) \end{aligned}$

_Example: Probability based on a graph_

```python
import graphviz
from sklearn.tree import export_graphviz
from matplotlib import pyplot as plt

tree="""
digraph Box {
a->b 
a->c 
b->c
b->d
c->e
}
"""
graphviz.Source(tree)
```
![graphviz](/images/2021/graph.png)


$P(\mathrm{a}, \mathrm{b}, \mathrm{c}, \mathrm{d}, \mathrm{e})=P(\mathrm{a}) P(\mathrm{b} \mid \mathrm{a}) P(\mathrm{c} \mid \mathrm{a}, \mathrm{b}) P(\mathrm{d} \mid \mathrm{b}) P(\mathrm{e} \mid \mathrm{c})$


## Notations of Random Variables

In the literature to denote a RV all these notations are acceptable:

* $\mathrm X$, or
* $X$
* $\mathrm x$

**Example**: _How to denote random variable $X$ has $k$ possible values?_

Answer:
$\{x_i\}_{i=1}^k$


The probability distribution of a discrete random variable is described by a list of probabilities associated with each of its possible values. 

This list of probabilities is called a probability mass function (PMF).

**Example**: _Sum of all probabilities should add to 1_

$P(X = red) = 0.3, P(X = yellow) = 0.45, P(X = blue) = 0.25$.

Each probability in a probability mass function is a value greater than or equal
to 0. The sum of probabilities equals 1.