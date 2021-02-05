---
published: true
layout: post
title: KL Divergence | Relative Entropy
permalink: /kl-divergence
---
- [Terminology](#terminology)
- [What is KL divergence really](#what-is-kl-divergence-really)
- [KL divergence properties](#kl-divergence-properties)
- [KL intuition building](#kl-intuition-building)
- [OVL of two univariate Gaussian](#ovl-of-two-univariate-gaussian)
- [Express KL divergence via Cross Entropy minus Entropy](#express-kl-divergence-via-cross-entropy-minus-entropy)


## Terminology

Kullback-Leibler Divergence (**KL Divergence**) know in statistics and mathematics is the same as **relative entropy** in machine learning and Python Scipy.

Let's start with the Python implementation to calculate the relative entropy of two lists:

```python
p=[0.2, 0.3, 0.5]
q=[0.1, 0.6, 0.3]   

def kl(a, b):
    '''
    numpy formula to calculate the KL divergence

    Parameters:
      a: pd of the random variable X
      b: pd of the random variable X
    Output:
      kl score always positive or 0 in case a=b)
    '''
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
k = kl(p,q)
print(k)
```
Output:
```
0.18609809382700082
```

## What is KL divergence really


We start by referring the basics:

Probability should add to 1

$\begin{aligned}\int_{-\infty}^{\infty} p(x) \mathrm{d} x =1\end{aligned}$

Expected value of the distribution $p$:

$\begin{aligned}\mu = \int_{-\infty}^{\infty} x p(x) \mathrm{d} x \end{aligned}$

Standard deviation formula, uses the expected value $\mu$.
 
$\begin{aligned}\sigma^{2} =\int_{-\infty}^{\infty}(x-\mu)^{2} p(x) \mathrm{d} x \end{aligned}$


KL divergence answers the question how different are two probability distributions $p$ and $q$ at any point $x$. 

$\begin{aligned}D_{KL} (q \| p)=\int_{-\infty}^{\infty} q(x) \log \frac{q(x)}{p(x)} d x\end{aligned}$

Recall that the $q(x)$ and $p(x)$ are probabilities and that the Shannon information is defined as:

$I(x) = -\log_2 q(x)$

This means KL is the **expected value of information** ratio.

$\begin{aligned}D_{KL}(q \| p) = - \mathbb{E}_{q}\left[-\log \frac{q}{p}\right]\end{aligned}$

## KL divergence properties

<div>

$\begin{array}{l}
\text { 1. } D_{KL}(q \| p) \neq D_{KL}(p \| q) \\
\text { 2. } D_{KL} \mathcal{L}(q \| q)=0 \\
\text { 3. } D_{KL} \mathcal{L}(q \| p) \geq 0
\end{array}$
</div>

The last property is easy to prove thanks to the Jensen's inequality for concave functions and logarithm is a concave function (the function is concave if it's second derivative is negative).

<div>

$\begin{aligned}-D_{KL}(q \| p) &=\mathbb{E}_{q}\left[-\log \frac{q}{p}\right]=\mathbb{E}_{q}\left[\log \frac{p}{q}\right] \\ & \leq \log \left[\mathbb{E}_{q} \frac{p}{q}\right]=\log \int q(x) \frac{p(x)}{q(x)} d x=0 \end{aligned}$
</div>
## KL intuition building

Now let's compare KL divergence of two Gaussian distributions:

$f(x)=\large \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{2}}$





We make the Python program to draw this PDF.

```python
%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 100

def gaussian_pdf(mu, sigma, x):
    '''
    Gaussian pdf
    Parameters:
        mu: mean
        sigma: standard deviation
    Output:
        calculated PDF value    
    '''
    
    'coeficient'
    c = 1/(sigma * np.sqrt(2*math.pi))
    'exponent'
    e = -0.5  * ((x-mu)/sigma)**2
    
    return c*np.exp(e)
    
gaussian_pdf(0,1,0)   
# 0.3989422804014327
```

First plot:
```python
x = np.arange(-5., 5., 0.1)
plt.plot(x,gaussian_pdf(0,1,x),color='r') # mu=0, sigma=1
plt.plot(x,gaussian_pdf(1,1,x),color='g') # mu=1, sigma=1
```

![gauss 1](/images/2021/02/gauss1.png)

Second plot:
```python
x = np.arange(-5., 5., 0.1)
plt.plot(x,gaussian_pdf(0,10,x),color='r') # mu=0, sigma=10
plt.plot(x,gaussian_pdf(1,10,x),color='g') # mu=1, sigma=10
```

![gauss 2](/images/2021/02/gauss2.png)

To compute the KL divergence between two Gaussian univariate functions we have the formula:

$$\begin{aligned} D_{KL}(p \| q) &=-\int p(x) \log q(x) d x+\int p(x) \log p(x) d x \\ &=\frac{1}{2} \log \left(2 \pi \sigma_{2}^{2}\right)+\frac{\sigma_{1}^{2}+\left(\mu_{1}-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}-\frac{1}{2}\left(1+\log 2 \pi \sigma_{1}^{2}\right) \\ &=\log \frac{\sigma_{2}}{\sigma_{1}}+\frac{\sigma_{1}^{2}+\left(\mu_{1}-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}-\frac{1}{2} \end{aligned}$$


**Calculation:**

In the first case:
$\mathcal N(0,1)$ and $\mathcal N(1,1)$

$D_{KL}(p\|q)=0.5$

Second case:
$\mathcal N(0,10)$ and $\mathcal N(1,10)$

$D_{KL}(p\|q)=0.005$

> Note in the second case the divergence is smaller.

## OVL of two univariate Gaussian 

We are building our KL divergence intuition further.

If we create a program in R to solve the OVL (overlapping coefficient) of the two Gaussian PDFs:

```r
min.f1f2 <- function(x, mu1, mu2, sd1, sd2) {
    f1 <- dnorm(x, mean=mu1, sd=sd1)
    f2 <- dnorm(x, mean=mu2, sd=sd2)
    pmin(f1, f2)
}

mu1 <- 0;    sd1 <- 1
mu2 <- 1;    sd2 <- 1

xs <- seq(min(mu1 - 3*sd1, mu2 - 3*sd2), max(mu1 + 3*sd1, mu2 + 3*sd2), .01)
f1 <- dnorm(xs, mean=mu1, sd=sd1)
f2 <- dnorm(xs, mean=mu2, sd=sd2)

plot(xs, f1, type="l", ylim=c(0, max(f1,f2)), ylab="density")
lines(xs, f2, lty="dotted")
ys <- min.f1f2(xs, mu1=mu1, mu2=mu2, sd1=sd1, sd2=sd2)
xs <- c(xs, xs[1])
ys <- c(ys, ys[1])
polygon(xs, ys, col="gray")

### only works for sd1 = sd2
SMD <- (mu1-mu2)/sd1
2 * pnorm(-abs(SMD)/2)

### this works in general
integrate(min.f1f2, -Inf, Inf, mu1=mu1, mu2=mu2, sd1=sd1, sd2=sd2)
```

![gauss 2](/images/2021/02/OVL1.png)

In here the OVL coefficient will be 0.617

![gauss 2](/images/2021/02/OVL2.png)

In here the OVL coefficient will be 0.96

> We should conclude that OVL coefficient may provide some intuition on the KL divergence.


## Express KL divergence via Cross Entropy minus Entropy

The relation is this:

$D_{KL}(p \|q) = H(q, p) - H(p)$


The upper equation holds for the discrete case where:

$\begin{aligned} H(p,q) = -\sum_x p\log q\end{aligned}$

$\begin{aligned} D_{KL}(p \| q) = \sum_{x} p\log {\frac{p}{q}} \end{aligned}$