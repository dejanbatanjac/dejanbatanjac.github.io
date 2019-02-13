---
published: false
---
Certainly one of the biggest update of Pytorch was 0.4 version where:

`torch.Tensor` become `torch.autograd.Variable`

This is a class capable of tracking history and behaves like the old Variable class.
This means we don't need Variable wrapping anymore in our code.


