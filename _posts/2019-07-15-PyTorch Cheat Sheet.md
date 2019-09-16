---
published: true
layout: post
title: PyTorch Cheat Sheet
---

>### Tensor creators

    import torch
    # by data
    t = torch.tensor([1., 1.])
    # by dimension
    t = torch.zeros(2,2)
    t = torch.ones(2,2)
    t = torch.empty(2,2)
    t = torch.rand(2,2)
    t = torch.randn(2,2)
    t = torch.arrange(1,10,0.2)

>### Concat and stack

    t = torch.tensor([1., 1.])
    c = torch.cat([t,t])
    s = torch.stack([t,t])
    print(c.size())# torch.Size([4])
    print(s.size())# torch.Size([2, 2])


>#### When something is a leaf

    x = torch.Tensor([1,2])
    print(x)
    print(x.is_leaf) # True
    y = x+1
    print(y.is_leaf) # True
    x = torch.tensor([1., 2. ], requires_grad=True)
    print(x)
    print(x.is_leaf) # True
    y = x+1
    print(y.is_leaf) # False

* Tensors that have requires_grad False will be leaf tensors by convention.
* For tensors that have requires_grad which is True, they will be leaf Tensors if they were created by the user. 
* This means that they are not the result of an operation and so grad_fn is None.


>### Assignment consideration

```
t=torch.tensor(1.)
a=t
print(t,a,id(t), id(a))
a.add_(1.)
print(t,a,id(t), id(a))
t.add_(1.)
print(t,a,id(t), id(a))
t+=1
print(t,a,id(t), id(a))
a+=1
print(t,a,id(t), id(a))
a=a+1
print(t,a,id(t), id(a))
a.add_(1.)
print(t,a,id(t), id(a))
a=a.add(1.)
print(t,a,id(t), id(a))
t+=1
print(t,a,id(t), id(a))
t=t+1
print(t,a,id(t), id(a))
a+=1
print(t,a,id(t), id(a))
```


>### Comparison with NumPy

```
np.empty((5, 3)) 	        | torch.empty(5, 3)
np.random.rand(3,2)             | torch.rand(3, 2) 
np.zeros((5,3)) 	        | torch.zeros(5, 3)
np.array([5.3, 3]) 	        | torch.tensor([5.3, 3]) 
np.random.randn(*a.shape)       | torch.randn_like(a) 	
np.arange(16)                   | torch.range(0,15) 
```

>### nn.Module

```
class M(nn.Module):
    def __init__(self):
        super().__init__();
        self.linear = nn.Linear(1,1)
    def forward(self, x):    
        y = self.linear(x)
        return y

```

>### Print the module

```
module = M()
print(module)
```

>### Check inner modules

This `modules()` method should provide more info than `children()`.

    for i, _ in enumerate(model.modules()):
        print (i, _)
        if (isinstance(_, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear))):
            print(_)


>### Creating optimizer modules:

Adam, RMSProp, AdaGrad, SGD...

```
from torch.optim import *
o = Adam(model.parameters(), lr=lr)
```


>### Initialize optimizer with empty tensor and convert every param to param groups

```
optimizer = optim.SGD({torch.empty(0)}, lr=1e-2, momentum=0.9 )
optimizer.param_groups.clear()
for p in model.named_parameters():
    optimizer.param_groups.append({'params' ,p})
    #print(p[0],":",p[1].size() )
```

>### Creating loss functions NLLLoss, MSELoss, CrossEntropyLoss...

```
loss = torch.nn.MSELoss(size_average=Fase)
```

>### Using pre-trained models:

```
from torchvision.models import resnet18
r = resnet18()

# Similar for VGG, Resnet, DenseNet, Inception,...
```

>### Setting the model in train or eval mode:

```
model.train()
model.eval()
```

>### Condition based:

```
t[t<=9.8619e-03] = 0 # set where condition

t[True] = 0 # set all to 0:

(t==0).sum() # check number of tensors eq. 0:

(t<0).sum() # number of elements smaller than 0

(t>0).sum() # number of elements greater than 0
```

>### Creating the device on GPU:0:

```
device = torch.device('cuda',0)
```

>### Save and load a tensor:

```
# Save to binary file
x = torch.tensor([0, 1, 2, 3, 4])
torch.save(x, 'file.pt')
# reverse operation
t = torch.load('file.pt') 
```

>### Writing PyTorch tensor to a file:

```
t = torch.rand(3)
f = "output.txt"    
def tensorwrite(file, t, text="tensor"):
    with open(file, "a") as f:    
        f.write(f"\n{text} {t}")
        f.close()
        
tensorwrite(f, t)
```

>### Getting actual size of the model:

```
import torch 
import torchvision.models as models
vgg16 = models.vgg16(pretrained=False)

size = 0
for p in vgg16.parameters():
  size += p.nelement() * p.element_size()
print(size)
```


>### addcdiv_

    import torch
    x = torch.Tensor([1., 3.])
    y = torch.Tensor([4., 4.])
    z = torch.Tensor([2., 4.])

    x.addcdiv_(2, y, z)
    x # tensor([5., 5.])

What just happened?

`x[0]` was `1`, but we added to that `2*y[0]/z[0]`, so we added `4`. Now the operation is in place so `x[0]` will end as `5`. 
Note: `addcdiv_` will do per element division.

