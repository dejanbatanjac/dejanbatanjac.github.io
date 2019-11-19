---
published: true
layout: post
title: Freezing layers (parameters) of a neural net
---

Freezing neural net parameters means not allowing parameters to learn. This is often needed if we use already trained models. There are two ways to freeze in PyTorch:

* setting `requires_grad` to `False`
* setting the learning rate `lr` to zero

Let's use resnet18 model to examine freezing layers.

## Using requires_grad

```
import torch
from torchvision.models import resnet18
model = resnet18(pretrained=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

for name, p in model.named_parameters():
    print("param name:", name, "requires_grad:", p.requires_grad)
``` 

Out:

```
param name: conv1.weight requires_grad: True
param name: bn1.weight requires_grad: True
param name: bn1.bias requires_grad: True
param name: layer1.0.conv1.weight requires_grad: True
...
```

As we can see all the parameters are able to learn, `requires_grad` is True.

You can set particular parameter `requires_grad` to `False` like this (both are the same):

```    
model.fc.weight.requires_grad_(False)
model.fc.weight.requires_grad = False
```

You can set all the parameters `requires_grad` to `False` this way:

```
for name, p in model.named_parameters():
     p.requires_grad = False
```    

The next code sets `requires_grad` to `True` for `conv1.weight` and `fc.weight` and `False` for the rest of parameters.

```
import torch
from torchvision.models import resnet18
model = resnet18(pretrained=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

for name, p in model.named_parameters():
    if name=='conv1.weight':
        p.requires_grad=True
    elif name=='fc.weight':
        p.requires_grad=True
    else:
        p.requires_grad=False

for name, p in model.named_parameters():        
    print("param name:", name, "requires_grad:", p.requires_grad)
```

The next code is a checker that we update the `model.fc.weight` parameter:

```
import torch
from torchvision.models import resnet18
model = resnet18(pretrained=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

for name, p in model.named_parameters():
        p.requires_grad=False

model.conv1.weight.requires_grad_(True)
model.fc.weight.requires_grad = True

bs=1
input = torch.rand(bs,3, 128,128)
target = torch.randint (1000, (1,))
model.train()

p1 = model.fc.weight.clone()
output = model(input)

loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(output, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()

p2 = model.fc.weight

print(torch.equal(p1,p2))
print(p1,p2)
```

## Zero learning rate

The `param_groups` is optimizer list of dictionaries. It usually has just a single group (item list) with the following keys:

    dict_keys(['params', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov'])

For example:

```
import torch
from torchvision.models import resnet18
model = resnet18(pretrained=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
for param_group in optimizer.param_groups:
    print(param_group['lr'])
```    

We can create two groups for the same model like this:

```
model = resnet18(pretrained=False)
optimizer.param_groups.clear()
optimizer.param_groups.append({'params' : model.conv1.parameters(), 'lr' : 0.3, 'name': 'model.conv1' })
optimizer.add_param_group({'params' : model.fc.parameters(), 'lr' : 0.4, 'name': 'model.fc' })
for param_group in optimizer.param_groups:    
    print(param_group['name'], param_group['lr'])    
```

Or we can create a new group for every param:

```
optimizer.param_groups.clear() 
for name, param in model.named_parameters():
    optimizer.add_param_group({'params' : param, 'lr' : 0.1, 'name':name})    
```

Let's check setting `fc.weight` learning rate to 0:

```
for p in optimizer.param_groups:
    if p['name']=='fc.weight':
        p['lr']=0

# for p in optimizer.param_groups:
#     print( p['name'], p['lr'])

input = torch.rand(bs,3, 256,256)
target = torch.randint (1000, (bs,))

model.train()
p1 = model.fc.weight.clone()

output = model(input)
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(output, target)
print(loss)
optimizer.zero_grad()
loss.backward()
optimizer.step()
p2 = model.fc.weight

print(torch.equal(p1,p2))  # True
```

This shows that p1 and p2 are the same, we haven't learned anything.

Here is the [gist](https://gist.github.com/dejanbatanjac/b5ed26a925c75514b8ab6d4e6a328e67) for this article.
