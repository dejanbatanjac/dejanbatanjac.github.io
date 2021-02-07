---
published: true
layout: post
title: The Power of Joint Probability
permalink: /joint-probability
---
- [What is joint probability of Random Variables](#what-is-joint-probability-of-random-variables)
- [If we know the joint probability...](#if-we-know-the-joint-probability)
- [Example with random dataset](#example-with-random-dataset)
- [Conclusion: MLE](#conclusion-mle)
- [Problem when we don't have enough rows](#problem-when-we-dont-have-enough-rows)
- [Solution: MAP](#solution-map)

## What is joint probability of Random Variables

Probabilistic approach to machine learning is this:

If we learn function $f:X→Y$ we actually learn $P(Y \mid X)$, where $Y$ is a target random variable and $X$ is a set of random variables: $X_1, ..., X_n$.

You may imagine $Y$ is a stock price, and $X$ are different factors we take. This way joint probability $Y$ of random variables $X$ is a function.

## If we know the joint probability...

Let's make a claim:

> If we know the joint probability distribution of random variables ${X_1, ... , X_n}$ we can easily answer specific joint or conditional probability questions on any subset of these variables...

If I know $P(X_1, X_2, ..., X_n)$ I can answer questions like:

$P(X_1 \mid X_2,  ... , X_n)$ or $P(X_1 \mid X_2)$ or $P(X_1, X_2)$, or $P(X_2 \mid X_1, X_3)$, or $P(X_1, X_2 \mid X_3, X_4)$ the list may be long.

## Example with random dataset
Let's create an example:

```python
import random
import pandas as pd
df = pd.DataFrame(columns=['gender', 'age', 'can_read', 'probability'])


for i in range(100):
    gender = random.choice([1, 0]) # male, female
    age = random.choice([3,4,5]) # years
    can_read = random.choice([0,1]) # cannot, can
    probability = random.random()
    new_row = {'gender':gender, 'age':age, 'can_read':can_read, 'probability':probability}
    df = df.append(new_row, ignore_index=True)
    
df = df.astype({"gender": int, "age": int, "can_read": int})
print(df)
```

Out:
```
    gender  age  can_read  probability
0        0    3         0     0.410984
1        1    4         1     0.186819
2        1    3         1     0.179636
3        0    5         0     0.682623
4        0    5         1     0.430356
..     ...  ...       ...          ...
95       1    3         0     0.938143
96       1    4         1     0.566523
97       0    5         1     0.813059
98       0    4         1     0.060813
99       0    3         0     0.340096

[100 rows x 4 columns]
```

Now we have 100 rows while we could easily have just 12 rows of probabilities. There are rows with the same `gender`, `age` and `can_read` but with the different `probability`.

We will average probability for those rows.

Here is how:

```python
df = df.groupby(['gender', 'age', 'can_read']).mean().reset_index()
df
```
Out:


|gender|age|can_read|probability|
|-|-|-|-|
|0|3|0|0.437074
|0|3|1|0.386973
|0|4|0|0.472777
|0|4|1|0.336604
|0|5|0|0.628947
|0|5|1|0.542314
|1|3|0|0.681251
|1|3|1|0.585545
|1|4|0|0.444165
|1|4|1|0.416327
|1|5|0|0.599410
|1|5|1|0.695121


Now this is better, but the probabilities need to add to 1. We will fix this:

```python
sum = df.probability.sum()
df.probability = df.probability/sum
df
```
Out:

|gender|age|can_read|probability|
|-|-|-|-|
|3|0|0|0.070196
|0|3|1|0.062149
|0|4|0|0.075930
|0|4|1|0.054060
|0|5|0|0.101011
|0|5|1|0.087098
|1|3|0|0.109411
|1|3|1|0.094041
|1|4|0|0.071335
|1|4|1|0.066864
|1|5|0|0.096267
|1|5|1|0.111639

Now we have normalized probabilities that add to 1.

Let's calculate $P(gender=1)$:

```
df[df.gender==1].probability.sum()
```
Out:

0.4504431645055146


Similar, we can calculate the probability of `gender` is female and `can_read` is True.

```
df[df.gender.eq(0) & df.can_read.eq(1)].probability.sum()
```

Out:

0.20330656361099614


So we can get the joint probability of any subset. What about conditional probabilities? 

We can use formula to calculate the conditional probability using the joint probability:

$P(Y \mid X) = \large \frac{P(X,Y)}{P(X)}$

So in case we need:

$P(gender=1 \mid age=3) = \large \frac{P(gender=1, age=3)}{P(age=3)}$

```python
p1 = df[df.gender.eq(1) & df.age.eq(3)].probability.sum()
p2 = df[df.age.eq(3)].probability.sum()
print(p1/p2)
```
Out:

0.6058784488672743

## Conclusion: MLE

So if we know joint probability distribution of random variables $X_1,...X_n$  then we can calculate conditional and joint probability distributions for any subsets of these variables.

Looks like we have "chiavi della città".

We can use this to solve any classification or regression problem.

Condition is we should have all the rows we need and all the probabilities. For instance, if we wish to predict if kids can read based on their age and gender we should set $P(can＿read \mid gender, age)$.

This is a basic setup of the **MLE (Maximum Likelihood Estimation)** technique. We just need to have enough data (number of rows).


## Problem when we don't have enough rows

What if we don't have enough data? Say we have 50 random variables. Many of those are non categorical. If all of the random variables would be categorical with exactly 2 categories each, this would be $2^{50}$ rows we need to cover all the probabilities we need.

But if we have one feature with 3 categories, such as year = {3,4,5} we already have this would be $3*2^{49}$ which is greater than $2^{50}$.

Where we have many different values for a column we may use inequalities, such as `age<=4`, `age>4`, and we could choose these inequality points similar as the ID3 algorithm can do for us (using Entropy and Information Gain).

Still $2^{50}$ is super big = 1.125.899.906.842.624


## Solution: MAP 
This is where **MAP (Maximum a Posteriori)** approximation can help. We use a _prior belief_ to express the end probability better. 

Actually we define some probability distribution for our random variable $Y$. We can try out different distributions and find the best one.