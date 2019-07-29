---
published: true
layout: post
title: The Impact of Matrix Multiplication
---
Wanted in here to present what happens when the matrix multiplication occurs several times.

I set the 50 repeat steps, which means 50 levels deep neural network.


```
import torch
import math
import matplotlib.pyplot as plt
def stat(t, p=True):
    m = t.mean()
    s = t.std()
    if p==True:
        print(f"MEAN: {m}, STD: {s}")
    return(m,s)
    
_m = []
_s = []

c = 100
r = 50# repeat steps
x = torch.randn(c)
m = torch.randn(c,c)#/math.sqrt(n)
stat(x)

for _ in range (0,r):
    x = m@x    
    _1, _2 = stat(x, False)
    _m.append(_1)
    _s.append(_2)
    

stat(x)

plt.plot(_m)
plt.plot(_s)
plt.legend(["mean","std"])
plt.show()
```        

At the very fist step we had:

`MEAN: -0.03876325488090515, STD: 0.879034161567688`

This means that `x = torch.randn(c)` line provided random normal initialization of our data with teh mean of ~0 and STD of ~1

And at the end this was:.

`MEAN: nan, STD: nan`

We can conclude even without the further digging that we need some method of fixing the mean and std of our product tensors somehow. 

This method is called the batch normalization.
...