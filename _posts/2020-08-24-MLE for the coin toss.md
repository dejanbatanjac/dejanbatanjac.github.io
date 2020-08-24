---
published: true
layout: post
title: MLE for the coin toss example
permalink: /mle-binomial
---

 
In the typical tossing coin example, with probability for the head equal to $p$ and tossing the coin $n$ times let's calculate the Maximum Likelihood Estimate (MLE) probability for the heads.

We know this is typical case of [Binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution){:ref="nofollow"} that is given with this formula:


$\operatorname{Bin}(k;n,p) = \binom{n}{k}p^k(1-p)^{n-k}$

( Read: $k$ is parametrized by $n$ and $p$)

We have:

$n=H+T$ is the total number of tossing, and $H=k$ is how many heads.

Leading to:

$\operatorname{Bin}(H;H+T,p) = \binom{H+T}{H}p^H(1-p)^{T}$

$\operatorname{Bin}(H;H+T,p)_{\operatorname{MLE}} = \underset{p}{\operatorname{arg\,max}} \binom{H+T}{H}p^H(1-p)^{T}$

$=\underset{p}{\operatorname{arg\,max}} \operatorname{log} \big[ \binom{H+T}{H}p^H(1-p)^{T} \big]$

$=\underset{p}{\operatorname{arg\,max}} \big[ \operatorname{log} \binom{H+T}{H} + \operatorname{log} p^H + \operatorname{log}(1-p)^{T} \big]$

$=\underset{p}{\operatorname{arg\,max}} \big[ H \operatorname{log} p + T \operatorname{log}(1-p) \big]$

We used `log` trick to gain numerical stability, and we removed the constant in this transformation process since it will not affect the `argmax`.

To get the MLE, and since this is the estimation we will find where first derivative of the is equal to zero:

$\large \frac{\partial [ H  \operatorname{log} p + T \operatorname{log}(1-p)]}{\partial p}=\small 0$

And this is true for:

$\large \frac{H}{p} = \frac{T}{1-p}$

So:

$\large p_{\small \text{MLE}} = \frac{H}{T+H}$

We could intuitively get the same conclusion, let's say we have some tossing events:


$\mathcal{T}=\\{h, h, h, t, t, h, t, t, t, h, t \\}$, tossing set

$n = T+H = 11$, and $H=5$

MLE is ${H \over {T+H}} = {5 \over 11}$.


<!--
#### Implementing `argmax` in Python

The `argmax` operator is simple to get the maximum argument. In Python code explaining the `argmax` would be like this:

```python
import numpy as np
  
mat = np.random.randint(50, size=(3, 4))
print(mat)   

print("Max element : ", np.argmax(mat)) 
print("Indices of Max elements (columns): ", np.argmax(mat, axis=0)) 
print("Indices of Max elements (rows): ", np.argmax(mat, axis=1)) 
```

Output:
```
[[13 21 48 49]
 [ 5 41 35  0]
 [31  5 30  0]]
Max element :  3
Indices of Max elements (columns):  [2 1 0 0]
Indices of Max elements (rows):  [3 1 0]
```
-->
