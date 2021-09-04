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
- [Probability of a single event (marginal probability)](#probability-of-a-single-event-marginal-probability)
- [Probability of two events](#probability-of-two-events)
  - [Union probability](#union-probability)
  - [Joint probability](#joint-probability)
  - [Conditional probability](#conditional-probability)
- [Bayes rule](#bayes-rule)
- [Chain rule of probabilities](#chain-rule-of-probabilities)
- [Different notation meaning](#different-notation-meaning)
 
## Basic probability notation
 
### Random variable
 
A *Random Variable* is a set of possible values from a random experiment. It should have associated **probability distribution** $P$. 
 
> <sub>It should also have **sample space** $S$ from where it takes the concrete values.</sub>
 
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
 
An event $e$ is a set of outcomes (one or more) from an experiment. An event can be:
 
* rolling a dice and getting 1
* getting head on coin toss
* getting an Ace from a deck of cards
 
Two events can be dependent or independent.
Two events can occur at the same time or no.
 
### Probability definition
 
Probability is the simple likelihood of an event occurring.
 
We use the term likelihood for something that already happened. We use the term probability for something that will happen.
 
So we can use likelihood for hypotheses, and probability to attach to possible results of the experiments.
 
Probabilities always sum to 1 as we know this is a fundamental property of a probability distribution.
 
This property is a direct consequence of the fact that the support for a probability distribution is mutually exclusive. The support is a set of possible values of a random variable having that distribution. 
 
For instance for the coin toss example we can either have the tail of the head outcomes. So the cardinality of the support is 2.
 
The likelihood is not a probability distribution, unless normalized. So the likelihood may not sum up to 1.
 
 
<!-- ## Probability vs. likelihood from the perspective of binomial trial
 
I find it excellent here to continue the explanation of the probability and likelihood difference from the perspective of the binomial trials. 
 
As we know the binomial trials (or Bernoulli trial) is a random experiment with exactly two possible outcomes:
 
* success
* failure
 
To describe the [binomial trial](https://programming-review.com/r/binomial) there are three items we are interested in:
 
* the number of successes 
* the number of independent trials
* the probability of a success
 
If we take the first item as unknown and other two as known we have a binomial distribution `B(n,p)`.
 
Here `n` is the number of trials, and `p` is the probability of success. The binomial distribution is probability distribution and it will add to 1.
 
Also if we take the probability `p` is known and the number of successes `k` is known we have negative binomial distribution `NB(r,p)`.
 
In negative binomial distribution the number of independent trials is unknown but this is still a probability distribution.
 
If we have the number of trials `n` and number of successes `k` we can calculate the ratio to get `p`.
 
If the probability `p` is what is missing and `n=10`, and `k=3` we can get `p=0.3`.
 
So what is the likelihood estimation then? Likelihood is a measure from hypothesis space. 
We need to provide hypothesis and spray some likelihood on them:
 
H1: `p=0.5` and 
H2: `p=0.1`.
 
Note that these hypotheses are estimations and they don't have actual meaning. 
 
To decide which hypothesis is more likely (so we use the term likelihood) we will calculate the ratio of the likelihoods.
 
This ratio of likelihoods is called Bayes factor. Calculating Bayes factor is a Bayesian alternative to classical hypothesis testing.
 
https://www.barelysignificant.com/post/bf/
 
https://www.youtube.com/watch?v=T-kMpA4z-7k
 
 
 
 
 -->
 
 
## Probability of a single event (marginal probability)
 
Probability of an event occurring $P(e)$ unconditionally. This means $P(e)$ is not conditioned on another event. We usually call marginal probability just probability.
 
**Example:** Newborn child is a boy
 
The probability that a newborn child is a boy is $P(boy) = 0.5$.
 
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
 
$P(h \mid e)$ can be expressed as:
 
Probability of event $e$ occurring, given that another event $h$ occurs.
 
$h$ is called the **hypothesis**, $e$ is the **evidence**.
 
> Event occurring may be by assumption, assertion or evidence.
 
In here we don't have the premise that the two events are _independent_. If $P(h \mid e) = P(h)$ then events $h$ and $e$ are independent.
 
> Events $h$ and $e$ **may or may not** happen simultaneously.
 
 $P(h \mid e) = \large {P(h \cap e) \over P(e)}$
 
**Example:** Given you pick a red card what is the probability that it is 5?
 
$P(5 \mid red) =\large {P(5 \cap red) \over P(red)} = {1/26 \over 1/2}=\frac{1}{13}$ 
 
## Bayes rule
The next formula is known as Bayes rule:
 
$P(h \mid e) = \large \frac {P(h) P(e \mid h)}{P(e)}$
 
$P(h \mid e) = \large \frac {P(h) P(e \mid h)}{P(e \mid h)P(h)+P(e \mid \overline h)P(\overline h)}$
 
Where:
 
* $P(h \mid e)$ is posterior probability
* $P(h )$ is prior probability
* $P(e \mid h) / P(e)$ is the likelihood ratio
* $P(e \mid h)$ is likelihood
 
> To get the Bayes formula just start with  conditional probability when $P(e_1, e_2) = P(e_2, e_1)$
 
 
## Chain rule of probabilities
 
Any joint probability distribution over many random variables may be decomposed
into conditional distributions over only one variable:
 
$P\left(x^{(1)}, \ldots, x^{(n)}\right)=P\left(x^{(1)}\right) \Pi_{i=2}^{n} P\left(x^{(i)} \mid x^{(1)}, \ldots, x^{(i-1)}\right)$
 
**Example:** Chain rule 1
<div>
 
$\begin{aligned} P(a, b, c) &=P(a \mid b, c) P(b, c), \\ P(b, c) &=P(b \mid c) P(c), \\ P(a, b, c) &=P(a \mid b, c) P(b \mid c) P(c) \end{aligned}$
</div>
 
**Example:** Chain rule 2
 
$P(a, b, c, d, e)=P(a \mid b,c,d,e) P(b \mid c,d,e) P(c \mid d, e) P(d \mid e) P(e)$
 
 
 
**Example**: _Sum of all probabilities should add to 1_
 
$P(\mathrm x=red) = 0.3$
 
$P(\mathrm x=yellow) = 0.45$
 
$P(\mathrm x=blue) = 0.25$.
 
The sum of **all** probabilities for a random variable $\mathrm x$ should add to 1.
 
## Different notation meaning
 
$P(x; y)$ is the density of the random variable $\mathrm x$ at the point $x$, where $y$ is a set of parameters. 
 
 
$P(x \mid y)$ is the conditional distribution of $\mathrm x$ given $\mathrm y$. It only makes sense if $\mathrm x$ and $\mathrm y$ are random variables.
 
 
$P(x,y)$ is the joint probability density of $\mathrm x$ and $\mathrm y$ at the point $(x,y)$. It only makes sense if $\mathrm x$ and $\mathrm y$ are random variables. 
 
$P(x\mid y,z)$ is similar to $P(x\mid y)$ but now $\mathrm z$ is a random variable.
 
Lastly $P(x\mid y;z)$ should mean that $z$ is a set of parameters, not a random variable.
 
