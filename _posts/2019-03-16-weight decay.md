---
published: false
layout: post
title: Weight Decay
---
Weight decay, as a normalization technique for neural nets deals with the fact of network complexity.
Traditionally, if you need to lower the network complexity you reduce the number of parameters.

However, you can still use a lot of parameters and penalise the complexity.
Let's create the model, the loss function and add to that sum the square of the parameters.

So we will have `(loss + sos).backward() # sos = sum of squares`  

In this case we  may zero all the parameters in the training phase. To prevent that we multiply the sum by some number called `wd` (weight decay): `(loss + wd * sos).backward()`. 

Typical value for the `wd` may be 0.1 or 0.01.

We calculate weights using w = w - lr * dL/dw
