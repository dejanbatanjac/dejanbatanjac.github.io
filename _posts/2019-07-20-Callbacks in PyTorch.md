---
published: true
layout: post
title: Callbacks vs. PyTorch hooks
---

Consider this simple example of a callback in Python:

```
from time import sleep

def batch_print(i):
    print("batch number ", i)
    
def calculation(x, cb=None):    
    for i in range(1, x+1):        
        sleep(1)
        if cb : cb(i)

calculation(4, batch_print)    
```
Out:
```
batch number  1
batch number  2
batch number  3
batch number  4
```

How would we do this in PyTorch using hooks?

```
from time import sleep

import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):        
        super().__init__()        
        self.l1 = nn.Linear(1,2)
        
    def forward(self, x):                      
        x = self.l1(x)
        return x

model = M()
model.train()
model.passedbatches=0

def batch_print(module, inp, outp):
    sleep(1)
    module.passedbatches +=1
    print("batch number", module.passedbatches)
    print("inp ", inp)
    print("outp ", outp)


model.register_forward_hook(batch_print)
for i in range(1,4):
    # simplified batches
    x = torch.randn(1)
    output = model(x)
```
Out:
```
batch number 1
inp  (tensor([-0.1330]),)
outp  tensor([-0.1543, -0.2368], grad_fn=<AddBackward0>)
batch number 2
inp  (tensor([-0.9720]),)
outp  tensor([-0.5334,  0.3447], grad_fn=<AddBackward0>)
batch number 3
inp  (tensor([-1.2069]),)
outp  tensor([-0.6396,  0.5076], grad_fn=<AddBackward0>)
```












