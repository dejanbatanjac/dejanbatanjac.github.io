---
published: false
---
PyTorch is exensible framework. We see that because many new frameworks used PyTorch as a foundation.
For instance [Fast.ai](https://github.com/fastai/fastai), took PyTorch as a foundation and built on it.

What kind of new functionality we can build:
* New modules
* New functions

I will provide some vanilla nn.Module first:
~~~
class Lambda(nn.Module):
    def __init__(self, f): super().__init__(); self.f=f
    def forward(self, x): return self.f(x)
~~~

We created with the last lines the new class Lambda (doing almost nothing) with the default constructor `__init__` and with the `forward` method.

Modules are the core of PyTorch. In modules we can club our parameters, layers, functions or even other modules. 

All modules live under `torch.nn` such as `torch.nn.Conv2d`, `torch.nn.Linear` etc.

--

Functions transform the data. We have several types of functions:
* common mathematical functions are implemented under `torch` such as `torch.log` or `torch.sum`
* neural network related functions under `torch.nn.functional`
* autograd operators under `torch.autograd.Function` imlementing the forward and backward functions. 

