---
published: true
title: What is new in PyTorch 1.0?
---
## PyTorch 0.4 version

Certainly the biggest update of Pytorch 0.4 version was when:

`torch.Tensor` become `torch.autograd.Variable`

This is a class capable of tracking history and behaves like the old Variable class.
This means we don't need Variable wrapping anymore in our code.

Note in this code how we used in place function `requires_grad_()` to set the tensor gradient tracker

~~~
import torch
tensor = torch.zeros(3, 4, requires_grad=False)
print(tensor.requires_grad)
#False
tensor.requires_grad_()
print(tensor.requires_grad)
#True
~~~

Note: `requires_grad` Tensor attribute is the central flag for autograd.
(Autograd is automatic backpropagation system in PyTorch)

What about the .data?
The `.data` attribute of the tensor would return the Tensor but with no autograd tracking capabilities as demonstrated in the next example.
~~~
data = tensor.data #new tensor but no autograd
print(data.requires_grad)
#False
~~~

From the perspective of backpropagation the `data` tensor is "dead". Anything that will happen on a `tensor` from the previous example will not reflect on a `data` tensor.

## PyTorch 1.0.0 version 

Imagine this simple program written in Python
~~~
import torch
model = torch.nn.Linear(5, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
prediction = model.forward(torch.randn(3, 5))
loss = torch.nn.functional.mse_loss(prediction, torch.ones(3, 1))
loss.backward()
optimizer.step()
~~~ 

Now, it is possible to write the same in C++
~~~
#include <torch/torch.h>
torch::nn::Linear model(5, 1);
torch::optim::SGD optimizer(model->parameters(), /*lr=*/0.1);
torch::Tensor prediction = model->forward(torch::randn({3, 5}));
auto loss = torch::mse_loss(prediction, torch::ones({3, 1}));
loss.backward();
optimizer.step();
~~~

Check [doc](https://pytorch.org/cppdocs) for more details.

--

Next, TorchHub is already trained repo of models.
TorchHub supports publishing trained models and weights) to a GitHub.
To do that we use `hubconf.py` file. 

Users can later load the trained models using the `torch.hub.load API`.

--

You can create empty tensor (no elements inside) with custom shape and dtype:

    tensor([], size=(0, 2, 4, 0), dtype=torch.float64)
    
Before, tensors with no elements were limted with shape (0,)

--
Note: For more details on PyTorch version 1.0.0 check [this](https://github.com/pytorch/pytorch/releases/tag/v1.0.0).