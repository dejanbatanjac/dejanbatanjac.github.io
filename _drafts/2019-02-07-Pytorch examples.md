---
published: false
---
Simplified we can create this PyTorch dictionary of things (PyTorch menu):

* Tensor
* Function
* Variable
* Computational graph
* Forward pass
* Backward pass
* Loss functions
* Optimizers
* Model


### Tensor

Here is how we can work with tensors.

~~~
import numpy as np
import torch

# tensor from memory
x = torch.Tensor(5, 3)
print(x)

# random
x = torch.randn(5, 3)
print(x)

# zeros
x = torch.zeros(5, 3)
print(x)

# ones
x = torch.ones(5, 3)
print(x)

# from numpy
x = torch.from_numpy(np.random.rand(5,2))
print(x)

# uniform
x = torch.Tensor(5, 3).uniform_(-1, 1)
print(x)

# clamp it further to (-.3,.3)
x = torch.clamp(x, -.3, .3)
print(x)
~~~

The output would be like this:
~~~
tensor([[ 5.6424e-36,  0.0000e+00,  4.4842e-44],
        [ 0.0000e+00,         nan,  1.8738e-01],
        [ 1.0791e-08,  1.0578e+21,  5.4424e+22],
        [ 1.3553e-05,  2.5966e+20,  1.0791e-08],
        [ 1.0500e-08,  1.3296e+22, -3.0000e-01]])
tensor([[ 0.6779,  2.8372, -1.9058],
        [-0.8177,  2.3343, -0.9823],
        [ 0.2937, -1.3743, -0.9953],
        [-0.1721, -0.6161,  1.0439],
        [-0.2200,  0.8456, -0.7462]])
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
tensor([[0.2207, 0.8334],
        [0.7960, 0.8807],
        [0.3208, 0.7189],
        [0.5759, 0.6944],
        [0.4807, 0.7277]], dtype=torch.float64)
tensor([[-0.5121,  0.1510,  0.9566],
        [-0.2695,  0.7197, -0.2671],
        [ 0.3743, -0.2660, -0.9023],
        [-0.3444, -0.3406, -0.9627],
        [-0.4546,  0.7440,  0.5265]])
tensor([[-0.3000,  0.1510,  0.3000],
        [-0.2695,  0.3000, -0.2671],
        [ 0.3000, -0.2660, -0.3000],
        [-0.3000, -0.3000, -0.3000],
        [-0.3000,  0.3000,  0.3000]])
~~~


The first example we created has `nan` and `0.0000e+00` inside. This speaks loudly that this method should not be used at all. Always use some of the following methods to generate the Tensor.


### Variable features

* holding tensors (.data attribute)
* requiring gradients
* volatility
* grad
* creator function

Inside variables you can access:
* raw tensor from the `.data` attribute
* gradient of the loss from the `.grad` attribute
* function from the `.grad_fn` attribute.

### ???
We create variables with `requires_grad = True` or  `requires_grad= False`
Some variables are volatile.



### The list of Loss functions

To execute backpropagation algorithm you need to have loss function in order to calculate the error. You can create your own loss function or use one from the list:

    L1Loss
    MSELoss
    CrossEntropyLoss
    CTCLoss
    NLLLoss
    PoissonNLLLoss
    KLDivLoss
    BCELoss
    BCEWithLogitsLoss
    MarginRankingLoss
    HingeEmbeddingLoss
    MultiLabelMarginLoss
    SmoothL1Loss
    SoftMarginLoss
    MultiLabelSoftMarginLoss
    CosineEmbeddingLoss
    MultiMarginLoss
    TripletMarginLoss
    
    
### The list of Optimizers



### Modules

The `torch.nn` package defines a set of classes such as `torch.nn.Module` class from where all PyTorch modules are defined.

By definition a module receives input tensors and computes output tensors, but it may also hold internal state such as tensors containing learnable parameters.


The example:
~~~
N, D_in, H, D_out = 64, 1000, 100, 10

X = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
~~~
The last model defines a sequence of layers `nn.Sequential` is a Module which contains other Modules, and applies them in sequence to produce its output. 

Next, we have a linear module computing output from input using a linear function, and holds internal Tensors for its weight and bias.

We pass it to the ReLU module, and again we pass it to the linera module but this time with D_out output size.














