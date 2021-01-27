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
  - [Union probability](#union-probability)
  - [Joint probability](#joint-probability)
  - [Conditional probability](#conditional-probability)
- [Chain rule of probabilities](#chain-rule-of-probabilities)
- [Different notation meaning](#different-notation-meaning)

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


**Example**: _How to denote random variable $X$ has $k$ possible values?_

Answer:
$\mathrm x = \{x_i\}_{i=1}^k$


The probability distribution of a discrete random variable $\mathrm x$ is described by a list of probabilities associated with each of its possible values $x_i$. 

Also for the discrete random variable $\mathrm x$ with the expression $P(x)$ we say probability that the event $x$ is true.

> In here $P$ is **pmf** (probability mass function).


### The Event

An event $e$ is a set of outcomes (one or more than one) from an experiment. An event can be:

* rolling a dice and getting 1
* getting head on coin toss
* getting an Ace from a deck of cards

Two events can be dependent or independent.
Two events can occur at the same time or no.

### Probability definition

Probability is simple likelihood of an event occurring.

We use the term likelihood for something that already happened. We use the term probability for something that will happen.


## Probability of two events
If we have two events we can define different probability types:

* union probability
* joint probability
* conditional probability


### Union probability

If events are _mutually exclusive_:

$P( e_1 \cup e_2) =P(e_1) + P(e_2)$

If events are not _mutually exclusive_:

$P( e_1 \cup e_2) =P(e_1) + P(e_2) -  P(e_1 \cap e_2)$ 

### Joint probability

We can use both notations:

$P( e_1 \cap e_2) = P(e_1, e_2)$

**Special conditions:**

> Two events $e_1$ and $e_2$ must happen at the **same time**. 
> 
> Two events $e_1$ and $e_2$ must be **independent**.

**Example:** 

Throwing two dice simultaneously.

**Notation:**

$P(x, y) = P(\mathrm x=x, \mathrm y=y) = P(x)*P(y)$

> Tip: You can change $x$ and $y$ with $e_1$ and $e_2$


### Conditional probability

Special case of _joint probability_ is **conditional probability**.

In here we don't have the premise that the two events are _independent_.

$P(e_1, e_2) = P(e_1)* P(e_2 \mid e_1)$,
$P(e_2, e_1) = P(e_2)* P(e_1 \mid e_2)$



**Notation:** _To grasp it easy we can use letters $h$ and $e$_

$P(h \mid e)$ can be expressed as:

Probability of an event $h$ given the knowledge that an event $e$ has already occurred.

$h$ is called the **hypothesis**, $e$ is the **evidence**.

The next formula is the Bayes rule:

$P(h \mid e) = \large \frac {P(h) P(e \mid h)}{P(e)}$

Where:

* $P(h \mid e)$ is posterior probability
* $P(h )$ is prior probability
* $P(e \mid h) / P(e)$ is the likelihood ratio
* $P(e \mid h)$ is likelihood

> To get the Bayes formula just start with  conditional probability when $P(e_1, e_2) = P(e_2, e_1)$


## Chain rule of probabilities

Any joint probability distribution over many random variables may be decomposed
into conditional distributions over only one variable:

$$P\left(\mathrm{x}^{(1)}, \ldots, \mathrm{x}^{(n)}\right)=P\left(\mathrm{x}^{(1)}\right) \Pi_{i=2}^{n} P\left(\mathrm{x}^{(i)} \mid \mathrm{x}^{(1)}, \ldots, \mathrm{x}^{(i-1)}\right)$$

**Example:** Chain rule

$\begin{aligned} P(\mathrm{a}, \mathrm{b}, \mathrm{c}) &=P(\mathrm{a} \mid \mathrm{b}, \mathrm{c}) P(\mathrm{b}, \mathrm{c}) \\ P(\mathrm{b}, \mathrm{c}) &=P(\mathrm{b} \mid \mathrm{c}) P(\mathrm{c}) \\ P(\mathrm{a}, \mathrm{b}, \mathrm{c}) &=P(\mathrm{a} \mid \mathrm{b}, \mathrm{c}) P(\mathrm{b} \mid \mathrm{c}) P(\mathrm{c}) \end{aligned}$

**Example:** Chain rule graph


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

## Different notation meaning

$P(x; y)$ is the density of the random variable $\mathrm x$ at the point $x$, where $y$ is set of parameters. 


$P(x \mid y)$ is the conditional distribution of $\mathrm x$ given $\mathrm y$. It only makes sense if $\mathrm x$ and $\mathrm y$ are random variables.


$P(x,y)$ is the joint probability density of $\mathrm x$ and $\mathrm y$ at the point $(x,y)$. It only makes sense if $\mathrm x$ and $\mathrm y$ are random variables. 

$P(x\mid y,z)$ is similar to $P(x\mid y)$ but now $\mathrm z$ is random variable.

Lastly $P(x\mid y;z)$ should mean that $z$ is a set of parameters, not a random variable.