---
published: false
---
Certainly one of the biggest update of Pytorch was 0.4 version where:

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


