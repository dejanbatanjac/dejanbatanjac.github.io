---
published: true
layout: post
title: PyTorch Cheat Sheet
---

>### Tensor creators

```
import torch
# by data
t = torch.tensor(1., 1.)
# by dimension
t = torch.zeros(2,2)
t = torch.ones(2,2)
t = torch.empty(2,2)
t = torch.rand(2,2)
t = torch.randn(2,2)
```

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

>###nn.Module

```
class M(nn.Module):
    def __init__(self):
        super().__init__();
        self.linear = nn.Linear(1,1)
    def forward(self, x):    
        y = self.linear(x)
        return y

```

>###Print the module

```
module = M()
print(module)
```

>### Creating optimizer modules:

Adam, RMSProp, AdaGrad, SGD...

```
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

>### Set multiple tensor values to 0 on condition:

```
t[t<=9.8619e-03] = 0
```

Set all values of a tensor to 0:
```
t[True] = 0
```

Check number of tensors values 0:
```
(t==0).sum()
```
Similar:
```
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