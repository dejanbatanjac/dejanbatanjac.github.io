---
published: false
---
Batch normalization is the regularization technique for neural networks presented for the first time in 2015 in this [paper](https://arxiv.org/abs/1502.03167).  

The paper epxlains the regularization effect, explains the improvements and *tries* to provide the clue why it works. 

## Achievement

Thanks to the batch norm for the first time the ImageNet exceeding the accuracy of human raters, or ML reached the accuracy of image classification over humans.


## How it works?

There are 4 three things important for the batch norm. 

* Apply BN to a single layer.
* Normalize the output from the layer activations
* Multiply normalized output by parameter `p1`
* Add to all of that the parameter `p2`

We can express this as: 

`y_ = n(f(w1, w2, ... wn, x) * p1 + p2`

Where `n` is the normalization function, `p1`, and `p2` are our scale and offset parameters and f is our function to create the output from the layer, and `y` are the activations.

`y = f(w1, w2, ... wn, x)`

...
