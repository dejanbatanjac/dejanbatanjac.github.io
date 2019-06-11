---
published: false
layout: post
title: Weight Decay
---
## Weight decay, as a normalization technique for neural nets is how you can manage network complexity.

Traditionally, if you need to lower the network complexity you reduce the number of parameters.

However, you can still use a lot of parameters and penalize the complexity.

Let's use a model where to the default loss function we add the sum the square of the parameters (sos).

So we will have:

    (loss + sos).backward()  

Now idea to minimize the loss is to multiply by zero all the parameters in the training phase. 

We introduce `wd` (weight decay):

    (loss + wd * sos).backward() 

Typical value for the `wd` may be `0.1` or `0.01`.

