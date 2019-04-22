---
published: true
---
Batch normalization is the regularization technique for neural networks presented for the first time in 2015 in this [paper](https://arxiv.org/abs/1502.03167).  

The paper explains the regularization effect, explains the improvements and tries to provide the clue why it works.

# Achievement

Thanks to the batch norm for the first time the ImageNet exceeding the accuracy of human raters, and we stepped the era where machine learning started to classify images better than humans.


# How it works?

There are five things important for the batch norm (BN):

* Apply BN to a single layer
* BN works on a single mini batch data
* Normalize the output from the layer activations
* Multiply normalized output by parameter weight
* Add to all of that the parameter bias

We can express this as:

`y_ = n(f(w1, w2, ... wn, x) * weight + bias`

Where `n` is the normalization function, `weight`, and `bias` are our scale and offset parameters, `f` is our function to create the output from the layer, `x` and `y` are the activations.

`y = f(w1, w2, ... wn, x)`

After the normalization we have the the mean of 0 and standard deviation of 1 for the single batch. Here is the example:

```
import torch
import torch.nn as nn

# affine=False means nothing to learn
m = nn.BatchNorm1d(10, affine=False)
input = 1000*torch.randn(3, 10)
print(input)
output = m(input)
print(output)
print(output.mean()) # should be 0
print(output.std()) # should be 1
```

If we dig into the code of the PyTorch class `_BatchNorm` we will find we are dealing with parameters weight and bias we can make learnable if we set `self.affine=True` :

```
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
```

But we can also see there are two more parameters `running_mean` and `running_var` that appears to be the shared for the every mini batch, that we can learn as well.
