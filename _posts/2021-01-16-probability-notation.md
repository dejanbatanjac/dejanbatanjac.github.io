---
published: true
layout: post
title: Probability notation
permalink: /probability-notation
---
- [Basic probability notation](#basic-probability-notation)
  - [Random variable](#random-variable)
  - [The Event](#the-event)
  - [Probability definition](#probability-definition)
- [Probability of two events](#probability-of-two-events)
  - [Joint probability](#joint-probability)
  - [Conditional probability](#conditional-probability)
- [Chain rule of probabilities](#chain-rule-of-probabilities)

## Basic probability notation

### Random variable

A *Random Variable* is a set of possible values from a random experiment. It should have associated probability distribution $P$.

In the literature to denote a Random Variable all these notations are acceptable:

* $\mathrm X$, or
* $X$
* $\mathrm x$

> We need to distinguish between algebra unknown variable $x$, and probability random variable $\mathrm x$.

The probability that $\mathrm x = x$ is denoted as $P( x )$. 

Sometimes we deﬁne a variable ﬁrst, then use $\sim$ notation to
specify which distribution it follows later: $\mathrm x ∼ P(x)$

### The Event

An event $E$ is a set of outcomes (1+) from an experiment. An event can be:

* rolling a dice and getting 1
* getting head on coin toss
* getting an Ace from a deck of cards

Two events can be dependent or independent.
Two events can occur at the same time or no.

### Probability definition

Probability is simple likelihood of an event occurring.

We use the term likelihood for something that already happened. We use the term probability for something that will happen.

**Example**: _How to denote random variable $X$ has $k$ possible values?_

Answer:
$\mathrm x = \{x_i\}_{i=1}^k$


The probability distribution of a discrete random variable $\mathrm x$ is described by a list of probabilities associated with each of its possible values $x_i$. 

## Probability of two events
If we have two events we can define two probability types:

* joint probability
* conditional probability

### Joint probability

Conditions:

> Two events $E_1$ and $E_2$ must happen at the **same time**. 
> 
> Two events $E_1$ and $E_2$ must be *independent*.

*Example:* 
Throwing two dice simultaneously.

*Notation:*

$P(x, y) = P(\mathrm x=x, \mathrm y=y) = P(x)*P(y)$



### Conditional probability

*Notation:*

$P(H | E)$ can be expressed as:

Probability of an event $H$ given the knowledge that an event $E$ has already occurred.

$H$ is called the Hypothesis, $E$ is the Evidence.

The next formula is the Bayes rule:

$P(H \mid E) = \large \frac {P(H) P(E \mid H)}{P(E)}$

Where:

* $P(H \mid E)$ is posterior probability
* $P(H E)$ is prior probability
* $P(E \mid H) / P(E)$ is the likelihood ratio
* $P(E \mid H)$ is likelihood



## Chain rule of probabilities

Any joint probability distribution over many random variables may be decomposed
into conditional distributions over only one variable:

$P\left(\mathrm{x}^{(1)}, \ldots, \mathrm{x}^{(n)}\right)=P\left(\mathrm{x}^{(1)}\right) \Pi_{i=2}^{n} P\left(\mathrm{x}^{(i)} \mid \mathrm{x}^{(1)}, \ldots, \mathrm{x}^{(i-1)}\right)$

_Example:_

$\begin{aligned} P(\mathrm{a}, \mathrm{b}, \mathrm{c}) &=P(\mathrm{a} \mid \mathrm{b}, \mathrm{c}) P(\mathrm{b}, \mathrm{c}) \\ P(\mathrm{b}, \mathrm{c}) &=P(\mathrm{b} \mid \mathrm{c}) P(\mathrm{c}) \\ P(\mathrm{a}, \mathrm{b}, \mathrm{c}) &=P(\mathrm{a} \mid \mathrm{b}, \mathrm{c}) P(\mathrm{b} \mid \mathrm{c}) P(\mathrm{c}) \end{aligned}$

*Example:* Chain rule graph


$P(\mathrm{a}, \mathrm{b}, \mathrm{c}, \mathrm{d}, \mathrm{e})=P(\mathrm{a}) P(\mathrm{b} \mid \mathrm{a}) P(\mathrm{c} \mid \mathrm{a}, \mathrm{b}) P(\mathrm{d} \mid \mathrm{b}) P(\mathrm{e} \mid \mathrm{c})$

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







**Example**: _Sum of all probabilities should add to 1_

$P(\mathrm x=red) = 0.3$

$P(\mathrm x=yellow) = 0.45$

$P(\mathrm x=blue) = 0.25$.

The sum of **all** probabilities for random variable $\mathrm x$ should add to 1.

