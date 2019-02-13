---
published: true
---
PyTorch is an extensible framework. We see that because many new things grow on PyTorch.

For instance, and probable the most obvious one: [Fast.ai](https://github.com/fastai/fastai), took PyTorch as a foundation.

What kind of new functionality we can build:
* New modules
* New functions

I will provide some vanilla `nn.Module` first:
~~~
class Lambda(nn.Module):
    def __init__(self, f): super().__init__(); self.f=f
    def forward(self, x): return self.f(x)
~~~

We created with the last lines the new class Lambda (doing almost nothing) with the default constructor `__init__` and with the `forward` method. This is all we need to create a new module:
* a constructor `__init__()`
* a `forward()` method

Inside PyTorch modules we can club our parameters, layers, functions or even other modules. 
All modules live under `torch.nn` such as: `torch.nn.Conv2d`, `torch.nn.Linear` etc.

--
Functions are another way to create new things in PyTorch. We have several types of functions:
* common mathematical functions are implemented under `torch` such as `torch.log` or `torch.sum`
* neural network related functions under `torch.nn.functional`
* autograd operators under `torch.autograd.Function` implementing the forward and backward functions.

The last kind of functions mentioned (autograd) allow us to customize PyTorch, introducing the new autograd functionality. Here is the example creating the new autograd:
~~~
class MyReLU(torch.autograd.Function):
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return x.clamp(min=0)

  def backward(ctx, grad_output):
    x, = ctx.saved_tensors
    grad_x = grad_output.clone()
    grad_x[x < 0] = 0
    return grad_x
~~~

Note how we defined both `forward()` and `backward()` functions being part of the MyReLU class inherited from `torch.autograd.Function`.

--
## nn package, and what is a module?

We mentioned previously the `torch.nn` package in PyTorch. The `nn` package defines a set of Modules, which are roughly equivalent to neural network layers. 

A Module is a unit that receives **input tensors** and computes **output tensors**.

Module also may hold ld internal state which are tensors containing learnable parameters.

In PyTorch the `nn` defines a set of useful loss functions that are commonly used when training neural networks. The Mean Squared Error (MSE) as probable the most used/common loss function.

Check [here](https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html) all the loss functions available. Every new PyTorch version some new loss function my be added.

Note: By definition every loss function is also a module in PyTorch.

## Optimizers

If you don't plan to manually set the tensor operations, and update the weights of your PyTorch model, you need simple optimization algorithms or **optimizers** such as:
* AdaGrad 
* RMSProp 
* Adam 
* SGD ...

The `torch.optim` package in PyTorch abstracts the idea of an optimization algorithm.

Most commonly optim methods are already there but you may add the custom optim algorithms.


