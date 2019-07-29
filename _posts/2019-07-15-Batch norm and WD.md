---
published: true
layout: post
title: Batch Norm and WD
---
To understand the idea of Batch Normalization we need to invest into several basic things:

* Idea of tensor normalization
* std, variance, and mean.
* the `lerp` function
* the running mean and variance
* PyTorch parameters and buffers
* PyTorch batches

### What does it mean to normalize the tensor? 

It simple means to subtracts the mean from the tensor and and to divide it by std.

### What is lerp?

<!-- Because `lerp` uses exactly the opposite momentum Batch norm uses that same momentum.

0.1 momentum in `lerp` (aka Batch norm) is the same as 0.9 momentum in Adam or other optimizer.
```
 |  Args:
 |      num_features: :math:`C` from an expected input of size
 |          :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
 |      eps: a value added to the denominator for numerical stability.
 |          Default: 1e-5
 |      momentum: the value used for the running_mean and running_var
 |          computation. Can be set to ``None`` for cumulative moving average
 |          (i.e. simple average). Default: 0.1
 |      affine: a boolean value that when set to ``True``, this module has
 |          learnable affine parameters. Default: ``True``
 |      track_running_stats: a boolean value that when set to ``True``, this
 |          module tracks the running mean and variance, and when set to ``False``,
 |          this module does not track such statistics and always uses batch
 |          statistics in both training and eval modes. Default: ``True``
 |  
 |  Shape:
 |      - Input: :math:`(N, C)` or :math:`(N, C, L)`
 |      - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
 ```
 
...

```
for name, buf in model.bn.named_buffers():
    print(name)
# running_mean
# running_var
# num_batches_tracked    
```
...

Only if `affine=True`.
```
for name, param in model.bn.named_parameters():
    print(name)
# weight
# bias
```
 -->

Here is the example of the `torch.lerp` function (linear interpolation function). This function will take two tensors `start` and `end` and create third tensor based on the first taking in account also the momentum value.

```
start = torch.arange(1., 5.) #tensor([1., 2., 3., 4.]) 
end = torch.empty(4).fill_(10) #tensor([10., 10., 10., 10.])
start.lerp_(end, 0.1) #tensor([1.9000, 2.8000, 3.7000, 4.6000])
```
As you may see the momentum value of `0.1` in here means take the 90% of the start value and 10% of the new value to create the third tensor.

This is why the first value of the third tensor is:
`1.9000 = 0.9 * 1. + 0.1 * 10`

Things get really interesting for lerp, 

Next, let's examine the important ingredient for the batch norm, the mean with the keep dimension:

```
t = torch.rand(1,2,3,4)
print(t)
m = t.mean((0,2,3), keepdim=True)
print("\nkeep", m, m.size())

m2 = t.mean((0,2,3), keepdim=False)
print("\ndon't keep", m2, m2.size())
```

If we don't keep the dimension we get the output of size `[2]`, else we get the output of size `[1, 2, 1, 1]` and in general case the output of size `[1, n, 1, 1]` if our `t` second dimension would be `n`, so it can broadcast nice.

```
tensor([[[[0.2090, 0.5522, 0.5752, 0.2166],
          [0.3513, 0.7169, 0.0956, 0.2433],
          [0.8815, 0.0304, 0.5402, 0.2191]],

         [[0.6097, 0.0666, 0.4091, 0.6735],
          [0.8864, 0.8898, 0.1764, 0.6285],
          [0.8523, 0.4908, 0.0480, 0.3505]]]])

keep tensor([[[[0.3859]], [[0.5068]]]]) torch.Size([1, 2, 1, 1])

don't keep tensor([0.3859, 0.5068]) torch.Size([2]).
```
To confirm this, the mean of the first 12 elements is : `0.3859`.
```
t = torch.tensor([0.2090, 0.5522, 0.5752, 0.2166, 0.3513, 0.7169, 0.0956, 0.2433, 0.8815, 0.0304, 0.5402, 0.2191])
print(tt)
print(tt.mean()) # tensor(0.3859)
```



