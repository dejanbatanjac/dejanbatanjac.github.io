---
published: true
layout: post
title: Creating ResNet18 from scratch
permalink: /resnet18
---

Recently I made some ResNet18 from scratch so I could modify it. Before I showed what is [inside ResNets](https://dejanbatanjac.github.io/2019/09/17/Resnet-inside.html) but in low detail.

## Few facts

There are several popular models: 
* ResNet18
* ResNet34
* ResNet50
* ResNet101

For ResNet18 and ResNet34 we use basic blocks, and for ResNet50 and ResNet101 we use bottleneck blocks.


We also have identity blocks and skip connection blocks. The difference is `ResIdentity` blocks have two and `ResSkip` blocks have three convolutions inside.

```python

import torch
import torch.nn as nn

class ResIdentity(nn.Module):
  # so called identity block with empty skip connection
  def __init__(self, ni): # in out channels
    super().__init__() 
    self.conv1 = nn.Conv2d(ni, ni, kernel_size=3, stride=1, padding=1, bias=False) 
    self.bn1 = nn.BatchNorm2d(ni)    
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(ni, ni, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(ni)    

  def forward(self, x):
    identity = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = x + identity
    x = self.relu(x)
    return x 


class ResSkip(nn.Module):
  # Tiny conv skip connection
  def __init__(self, ni, no): # in channels, out channels
    super().__init__() 
    self.conv1 = nn.Conv2d(ni, no, kernel_size=3, stride=2, padding=1, bias=False) 
    self.bn1 = nn.BatchNorm2d(no)    
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(no, no, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(no)    
    self.skip_conv1 = nn.Conv2d(ni, no, kernel_size=1, stride=2, padding=0, bias=False) 
    self.skip_bn1 = nn.BatchNorm2d(no)

  def forward(self, x):
    skip = self.skip_conv1(x)
    skip = self.skip_bn1(skip)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = x + skip
    x = self.relu(x)
    return x

```

What we use to call basic blocks have two variants, first made by N identity blocks.

```python
class NIdentityBlocks(nn.Module):  
  def __init__(self, ni, repeat=2): 
    super().__init__()       
    self.block = nn.Sequential()
    for _ in range(repeat):
      self.block.add_module(f"ResIdentity{_}",  ResIdentity(ni))
    
  def forward(self, x):    
    x = self.block(x)    
    return x
```

Second made by skip blocks followed by N identity blocks.

```python
class SkipAndNIdentityBlocks(nn.Module):  
  def __init__(self, ni, no, repeat=2): 
    super().__init__()        
    self.block = nn.Sequential()
    self.block.add_module("ResSkip", ResSkip(ni, no))
    for _ in range(repeat-1):
      self.block.add_module(f"ResIdentity{_}",  ResIdentity(no))
    
  def forward(self, x):
    x = self.block(x)
    return x

```

The idea of ResNet head and tail corresponds to the encoder and decoder.
```python
class ResNetHead(nn.Module):
    def __init__(self, ni, no):
        super().__init__()        
        self.conv = nn.Conv2d(ni, no, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(no)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class ResNetTail(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.lin = nn.Linear(ni, no)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        return x
```

Lastly the full ResNet is a composition. The few things we define is the number of inputs (usually 3) and the number of outputs (usually 1000).

The channels 64 is the initial number of planes (channels) and in the end we have the 512 channels. All ResNet architectures will have these l0, l1, l2, l3 layers doubling the output channels by factor 2.

```python
class ResNet(nn.Module):

    def __init__(self, ni, no, repeat):
        super().__init__()
        self.head = ResNetHead(ni, 64)
        self.l0 = NIdentityBlocks(64, repeat[0])
        self.l1 = SkipAndNIdentityBlocks(64, 128, repeat[1])
        self.l2 = SkipAndNIdentityBlocks(128, 256, repeat[2])
        self.l3 = SkipAndNIdentityBlocks(256, 512, repeat[3])
        self.tail = ResNetTail(512, no)
        
    def forward(self, x):
        x = self.head(x)
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.tail(x)
        return x
```

## The Check

```python
i = torch.rand(85, 3, 32,32)
my_resnet18 = ResNet(3,1000, [2,2,2,2])
o = my_resnet18(i)
# print(o.size())  
# print(my_resnet18)
nparams = sum(p.numel() for p in my_resnet18.parameters())
print(nparams) # 11689512
```

You will find `my_resnet18` has 11689512 parameters.
This is the same as in PyTorch.

```python
import torchvision.models as models
resnet18 = models.resnet18(False)
nparams = sum(p.numel() for p in resnet18.parameters())
print(nparams) # 11689512
```


## Initialization tip

We haven't initialized the conv layers, but to do that I would use this function:

```
def reset_parameters(self):
      init.kaiming_uniform_(self.weight, a=math.sqrt(3))
      if self.bias is not None:
          fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
          bound = 1 / math.sqrt(fan_in)
          init.uniform_(self.bias, -bound, bound)
```