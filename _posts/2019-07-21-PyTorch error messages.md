---
published: false
layout: post
title: Error messages
---

* When I tried to set the BN 1d on input tensor (bs, 784) (case of MNIST)
self.bn0 = nn.BatchNorm1d(100)
```
RuntimeError: running_mean should contain 784 elements not 100
```

* When batch norm uses wrong number of features
```
RuntimeError: running_mean should contain 250 elements not 150
```

* When you deal with tensors and you request the missing dimension.
```
torch.cat((torch.rand(1,2), torch.rand(1,2)), dim=2)
torch.cat((torch.rand(1,2,3), torch.rand(1,2,3)), dim=3)
torch.cross(torch.rand(1,2), torch.rand(1,2), dim=2)
torch.cross(torch.rand(1,2,3), torch.rand(1,2,3), dim=3)
```
You may get this error:
```
RuntimeError: dimension out of range (expected to be in range of [-2, 1], but got 2)
```
[-2, 1] in the next RTE means the same. dim=-2 is the same as dim=1, and dim=-1 is the same as dim=0.

* when...
```
RuntimeError: output with shape [250] doesn't match the broadcast shape [1, 250]
```

* when...
```
RuntimeError: output with shape [1, 250, 1, 1] doesn't match the broadcast shape [1, 250, 1, 250]
```

* When the model was on CPU and the data was on GPU
```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```


```
NameError: name 'F' is not defined
```
