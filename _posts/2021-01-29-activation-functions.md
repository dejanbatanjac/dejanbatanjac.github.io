---
published: true
layout: post
title: Activation functions
permalink: /activation-functions
---
- [Activation functions overview](#activation-functions-overview)
- [Identity](#identity)
- [Binary step](#binary-step)
- [ReLU](#relu)
- [Leaky ReLU](#leaky-relu)
- [Sigmoid](#sigmoid)
- [Tanh](#tanh)
- [GELU](#gelu)
- [Swish](#swish)
- [Softplus](#softplus)
- [Mish](#mish)

## Activation functions overview

Activation functions are applying an affine transformation combining weights and input features. They are by default nonlinear functions.

The most important question of machine learning is why do we need nonlinear activation functions?

**The purpose of the activation function is to introduce non-linearity into the network**.

Nonlinear means that the output cannot be reproduced from a linear combination of the inputs (affine transform).

_Without a nonlinear activation function neural network with any number of layers would behave just like a single-layer perceptron, because summing these layers would give you just another linear function._

The most important features to track for the specific activation functions are:

* function input range 
* output range 
* if function is monotonic
* what is the derivate of a function
* if derivate is monotonic


I created the very complete list of activation functions:

* Identity
* Binary step
* ReLU
* Sigmoid (logistic, or soft step) 
* Tanh 
* GELU
* Leaky ReLU
* Swish
* PReLU
* GLU
* Softplus
* Maxout
* ELU
* Mish
* ReLU6
* Hard Swish
* SELU
* Softsign 
* Shifted Softplus
* CReLU
* RReLU
* Hard Sigmoid
* SiLU
* KAF
* TanhExp
* SReLU
* modReLU
* Hermite 
* ARiA
* E-swish
* m-arcsinh
* PELU
* ELiSH
* HardELiSH
* SERLU
* nlsig
* Lecun's Tanh
* Hardtanh 
* ASAF

I will shortly cover some of the functions, at least some of them that are frequently used:

## Identity

The most simple (intriguing) function:

$f(x) = x$

Has a very simple derivate. When we deal with activation functions, usually we like to have a nice derivate for the activation function.

The problem with this function is the infinite range of values it may return. Usually we like $(-1,1)$ or $(0,1)$ ranges.

## Binary step

Defined super simple this is one of the most basic functions:

<div>

$f(x) =\left\{\begin{array}{ll}0 & \text { if } x<0 \\ 1 & \text { if } x \geq 0\end{array}\right.$
</div>

The problem with this function it is: derivate is not defined for $x=0$.

## ReLU

$f\left(x\right) = \max\left(0, x\right)$

Rectified Linear Units, or ReLUs, are the most common activation forms. Linearity in the positive dimension has the attractive property that it prevents non-saturation of gradients (contrast with sigmoid activations), although for half of the real line its gradient is zero.


## Leaky ReLU
<div>

$f(x) = \left\{\begin{array}{ll}0.01 x & \text { if } x<0 \\ x & \text { if } x \geq 0\end{array}\right.$
</div>

Leaky Rectified Linear Unit, or Leaky ReLU, is based on a ReLU, but it has a small slope for negative values instead of a flat slope. 

The slope coefficient is determined before training, (not during training). This type of activation function is popular where we we may suffer from sparse gradients, for example training GANs (Generative Adversarial Networks).


There is one modification of Leaky ReLU called parametrized Leaky ReLU:

<div>

$f(x) = \left\{\begin{array}{ll}\alpha x & \text { if } x<0 \\ x & \text { if } x \geq 0\end{array}\right.$
</div>


## Sigmoid


$f(x)=\Large \frac{1}{1+e^{-x}}$

Drawback: sharp damp gradients during backpropagation from deeper hidden layers to inputs, gradient saturation, and slow convergence.

Derivate of this function is:

$f(x)(1-f(x))$


## Tanh

$f\left(x\right) = \large \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$

The tanh function became preferred over the sigmoid function as it gave better performance for multi-layer neural networks. But it did not solve the vanishing gradient problem that sigmoids suffered, which was tackled more effectively with the introduction of ReLU activations.

Derivate: $f(x)' = 1-f(x)^2$

## GELU

$\begin{aligned} & f(x) = \frac{1}{2} x\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right) =& x \Phi(x) \end{aligned}$

where $X\sim \mathcal{N}(0,1)$.


The Gaussian Error Linear Unit, or GELU is $x\Phi(x)$, where $\Phi(x)$ the standard Gaussian cumulative distribution function. 

One can approximate the GELU with: $0.5x\left(1+\tanh\left[\sqrt{2/\pi}\left(x + 0.044715x^{3}\right)\right]\right)$ or $x\sigma\left(1.702x\right),$ but PyTorch's exact implementation is sufficiently fast such that these approximations may be unnecessary. 

GELUs are used in GPT-3, BERT, and most other Transformers.



## Swish

$f(x) = \Large \frac{x}{1+e^{-\beta x}}$

Here $\beta$ a learnable parameter. Nearly all implementations do not use the learnable parameter $\beta$, in which case the activation function is $x\sigma(x)$ ("Swish-1").

The function $x\sigma(x)$ is exactly the SiLU introduced from GELUs paper.



## Softplus 


$f(x) = \ln \left(1+e^{x}\right)$

It can be viewed as a smooth version of ReLU.
Ths function has an interesting derivate which is exactly the sigmoid function.

$f(x)=\Large \frac{1}{1+e^{-x}}$



## Mish

$f\left(x\right) = x\cdot\tanh{\text{softplus}\left(x\right)}$

where

$\text{softplus}\left(x\right) = \ln\left(1+e^{x}\right)$


