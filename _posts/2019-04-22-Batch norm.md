---
published: true
---
Batch normalization is the regularization technique for neural networks presented for the first time in 2015 in this [paper](https://arxiv.org/abs/1502.03167).  

The paper explains the regularization effect, explains the improvements and tries to provide the clue why it works.

# Achievement

Thanks to the batch norm for the first time the ImageNet exceeded the accuracy of human raters, and stepped the era where machine learning started to classify images better than humans (for the particular classification task).


# How it works in PyTorch?

There are few things important for the batch norm (BN):

* Apply BN to a single layer for every mini batch
* Normalize the output from the batch activations
* In PyTorch if we set`affine=True`
* *  Multiply normalized output by parameter called `weight` and add to that 
* * Add to all of that the parameter `bias`
* If we set `track_running_stats=True` in PyTorch
* * Running statistics will be calculated
* * BN output will be less bumpy 

Simplified (without using the running statistics) we can express this as:

$$y_ = n(f(w_1, w_2, ... w_n, x)) * weight + bias$$

Where $n$ is the normalization function, $weight$, and $bias$ are our scale and offset parameters, $f$ is our function to create the output from the layer, $x$ and $y$ are the activations.

$$y = f(w_1, w_2, ... w_n, x)$$


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

PyTorch class `_BatchNorm` explains clearly we use the parameters `weight` and `bias` if we set `self.affine=True` :

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

Previous code excerpt also shows two more buffers `running_mean` and `running_var` that are calculate every mini batch to to make the BN output less bumpy.

Couple things to cover from the previous code:

There is a concept of module parameter (`nn.Parameter`). Parameter is just a tensors limited to the module where it is defined. 
Usually it will be defined in the module constructor (`__init__` method).

`register_parameter` method in previous code will do some safe checks before set the parameter to `None`, meaning we will not learn the values of `weight` and `bias` if `self.affine` is not `True`.

Once we have module parameter defined, it will appear in the `module.parameters()`.

`register_buffer` is specific tensor that can go to GPU and that can be saved with the model, but it is not meant to be learned (updated) via gradient descent. Instead it is calculated at every mini batch step.

As you may noted in PyTorch we have the training time and the inference time. While we train (learn/fit) we will constantly update the `running_mean` and `running_var` with the every mini batch. In the inference time we will just use the values calculated and we will not alter the running mean and var.

Note: running mean and running variance, are statistical methods calculating the [moving average](https://en.wikipedia.org/wiki/Moving_average). What they essentially do you can spot from the image.

![IMG](/images/maverage.png)

Lastly, it is possible to use BN even if we set `affine=False` and `track_running_stats=False`. It will just work.

For completeness, batch norm is one of the four types of the regularization techniques.

![IMG](/images/batch1.png)

REF: https://arxiv.org/abs/1502.03167