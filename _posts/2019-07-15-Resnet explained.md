---
published: true
layout: post
title: Resnet simple explained
---


Normal convolution simplified
$$h_{t+1} = c(c(h_t)) $$

Resnet block simplified
$$h_{t+1} = c(c(h_t)) + h_t $$ 

Note the $+$ sign literally means the `sum` operation, not the concatenation.

We can express it other way also:
$$h_{t+1} - h_t = c(c(h_t))) $$
or 
$$h_{t+1} - h_t = R(h_t)$$

This part $ h_{t+1} - h_t $ is called the <b>residual</b> and that is why we have the Resnet name.

Important: Inside the Resnet block there is no stride and no max pooling layers, so we are dealing with the same dimensionality inside the block.

Typically the architecture of Resnet is not using the dropout also.

The architecture also means several Resnet blocks are setting at top of each other.


