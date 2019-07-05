---
published: true
layout: post
title: Softmax vs. Sigmoid functions
---

In Machine Learning, you deal with `softmax` and `sigimoid` functions often.
I wanted to provide some intuition when you should one, over the other.

Suppose you have predictions as the output from neural net.

![IMG](/images/ss1.png)

These are the predictions for cat, dog, cow, and zebra. They can be positive or negative (no `ReLU` at the end).

### Softmax

If we plan to find exactly one value we should use `softmax` function.
The character of this function is "there can be only one".

Note the out values are in the `B` column. Then for each `B` value $x$ we do create $e^x$ in column `C`.

What the `exp` function do it will do:
* it will make the predictions positive
* what ever was max, it will stand out as max

The `softmax` funciton:

$$ softmax( x_i ) =  {     e^{x_i} \over \sum_{j=1}^k { e^{x_j} } } $$

Can be literally expressed as take the exponent value and divide it by the sum of all other exponents (~`34` in the image). This will make one important feature of `softmax`, that the sum off all softmax values will add to 1.

Just by peaking the `max` value after the softmax we get out prediction. It is that easy, or the index of the prediction.

### Sigmoid

Things are different for the `sigmoid` function. This function can provide us with the top $n$ results based on the threshold.

If the threshold is e.g. `3` from the image you can find two results greater than that number. We use the following formula to evaluate the `sigmoid` function.

$$ sigmoid( x ) =  { e^{x} \over 1+ e^{x} } $$

Exactly, the feature of `sigmoid` is to emphasize multiple values, based on the threshold, and we use it for the multi-label classification problems.

### And in PyTorch...

In PyTorch you would use the `torch.nn.Softmax(dim=None)` layer compute `softmax` to an n-dimensional input tensor rescaling them so that the elements of the n-dimensional output tensor lie in the range [0,1] and sum to 1.
```
import torch.nn as nn
m = nn.Softmax(dim=0)
inp = torch.randn(2, 3)*2-1
print(inp)

out = m(inp)
print(out)
print(torch.sum(out))

# tensor([[-1.2928, -2.9990, -1.8886],
#         [ 0.1079, -3.6320, -1.6835]])
# tensor([[0.1977, 0.6532, 0.4489],
#         [0.8023, 0.3468, 0.5511]])
# tensor(3.)

```
Note you need to specify the dimension for `softmax`, which is `dim=0` in the previous example (dimension of columns). This is why the total sum will add to 3. since we have three columns.

But we can also functional version of `softmax`. The previous example can be rewritten as:

```
import torch.nn.functional as F
inp = torch.randn(2, 3)*2-1
print(inp)
out = F.softmax(inp, dim=0)
print(out)
print(torch.sum(out))

# tensor([[ 0.9096, -2.5876, -2.2403],
#         [-0.8566,  0.2757, -1.9268]])
# tensor([[0.8540, 0.0540, 0.4223],
#         [0.1460, 0.9460, 0.5777]])
# tensor(3.)
```

There is also a special 2d `softmax` that works on 4D tensors only, but you can always rewrite it using the regular `F.softmax`.
```
m = nn.Softmax2d()
inp = torch.randn(1, 3, 2, 2)
out = m(inp)
out2 = F.softmax(inp, dim=1)

print(torch.equal(out, out2)) #True
```

For the `sigmoid` function the things are quite clear.
```
inp = torch.randn(1,5)
print(inp)
print(F.sigmoid(inp))
# tensor([[-0.4010,  0.0468, -0.4071,  0.6252,  1.0899]])
# tensor([[0.4011, 0.5117, 0.3996, 0.6514, 0.7484]])
```











