---
published: true
layout: post
title: Impact of Weight Decay
---
I am using in here the Logistic Regression neural network. This is a single linear layer (nn.Linear in PyTorch).

<sub>That would be Dense layer in TensorFlow.</sub>

This neural network doesn't even have a single activation function (F.relu or similar)

The main reason to analyse LR is because it is simple, and can allow us to examine batch loss and impact of Weight Decay on BL.

I created the following example using the MNIST dataset in PyTorch:

```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import *
import torchvision

dl = DataLoader( torchvision.datasets.MNIST('/data/mnist', train=True, download=True), shuffle=False)

tensor = dl.dataset.data
tensor = tensor.to(dtype=torch.float32)
tr = tensor.reshape(tensor.size(0), -1) 
tr = tr/128 # tr = tr/255
targets = dl.dataset.targets
targets = targets.to(dtype=torch.long)

x_train = tr[0:50000-1]
y_train = targets[0:50000-1]
x_valid = tr[50000:60000-1]
y_valid = targets[50000:60000-1]

bs=64

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, drop_last=False, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

loaders={}
loaders['train'] = train_dl
loaders['valid'] = valid_dl


class M(nn.Module):
    'custom module'
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)
      
    def forward(self, xb):
        return self.lin(xb)

def probe(model, criterion, optimizer, bs, epochs, lr, wd_factor, color):
    
    losses=[]
    for epoch in range(0,epochs):
        train_loss = 0
        valid_loss = 0    
        print(f"Epoch {epoch}")

        model.train()
        for i, (data,target) in enumerate(loaders['train']):                
            optimizer.zero_grad()
            output = model(data)
            wd = 0.
            for p in model.parameters(): 
                wd += (p**2).sum()

            loss = criterion(output, target)+wd*wd_factor 
            train_loss += loss.item()
            loss.backward()            
            optimizer.step()
            if (i%1==0):
                #print(f"Batch {i}, loss {loss.item()}")
                losses.append(loss.item())

        model.eval()
        for i, (data,target) in enumerate(loaders['valid']):                
            output = model(data)
            loss = criterion(output,target)
            valid_loss += loss.item()        

        train_loss = train_loss/len(loaders['train'])
        valid_loss = valid_loss/len(loaders['valid'])        


        print(f"Train loss: {train_loss}")
        print(f"Validation loss: {valid_loss}")        
        print("wd_factor", wd_factor)
        plt.plot(losses, color)


# character 	color
# b 	blue
# g 	green
# r 	red
# c 	cyan
# m 	magenta
# y 	yellow
# k 	black
# w 	white        
```

The `probe` function is what I will be using to examine things.
I will be using two different optimizers, `Adam` and `SGD` that both have weight decay, but I also implemented weight decay to test as a separate.

This is the simplest form of the optimizers (no momentum, no wd).
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


Our criterion will be `nn.CrossEntropyLoss()`.

## Impact of the learning rate
```
criterion = nn.CrossEntropyLoss()

bs=64
epochs = 1
wd_factor = 0.0

lr = 0.1
model = M()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
probe(model, criterion, optimizer, bs, epochs, lr, wd_factor, "r") #red

lr = 0.01
model = M()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
probe(model, criterion, optimizer, bs, epochs, lr, wd_factor, "g") #green

lr = 0.001
model = M()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
probe(model, criterion, optimizer, bs, epochs, lr, wd_factor, "b") #blue
```

![LSTM](/images/lreg1.png)

This image represents a single epoch. Note how the biggest learning rate (red) decreases the batch loss really fast, but then has a strong oscillation, comparing to the blue.

```
Epoch 0 #red
Train loss: 0.3989532570761945
Validation loss: 0.2992929560662825
wd_factor 0.0
Epoch 0 #green
Train loss: 0.6678904935603251
Validation loss: 0.40658352872993375
wd_factor 0.0
Epoch 0 #blue
Train loss: 1.4549870281420705
Validation loss: 0.9654966373986835
wd_factor 0.0
```

## Using WD 4e-3

The basic assumption is that weight decay can fix these oscillations.
I tried to understand the impact of `weight_decay=4e-3` on SGD.

For that I needed two subplots side by side:
```
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_ylim(0,2)
ax2.set_ylim(0,2)
fig.suptitle('3epochs no WD vs. 4e-3 WD')
fig.set_figheight(5)
fig.set_figwidth(15)
```

And I set 3 epochs area of comparison.
```
criterion = nn.CrossEntropyLoss()

bs=64
epochs = 3
wd_factor = 0.0

lr = 0.1
model = M()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
probe(ax1, model, criterion, optimizer, bs, epochs, lr, wd_factor, "r") #red


lr = 0.1
model = M()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=4e-3)
probe(ax2, model, criterion, optimizer, bs, epochs, lr, wd_factor, "c") #cian
```

![LSTM](/images/lreg2.png)

As you can note even though I tried really hard to find ideal WD factor, the benefit was almost none.

I tried in the next attempt to compared the two approaches, where I use SGD WD 4e-3 (black) and SGD WD 4e-3 together with custom WD 4e-3

```
lr = 0.1
wd_factor=0
model = M()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
probe(ax[0], model, criterion, optimizer, bs, epochs, lr, wd_factor, "k") #black


lr = 0.1
wd_factor=4e-3
model = M()
optimizer = torch.optim.SGD(model.parameters(), lr=lr,  weight_decay=4e-3)
probe(ax[1], model, criterion, optimizer, bs, epochs, lr, wd_factor, "m") #magenta
```

![LSTM](/images/lreg3.png)

As you may see, the results are pretty mach the same except that using WD really has sense when parameters are growing big from some reason.

However, there are other techniques to suppress the parameters to grow unreasonable, so WD is not so popular any more.


## Conclusion

WD is a regularization term that penalizes big weights.

When the weight decay coefficient is big, the penalty for big weights is also big, when it is small weights still may grow.

But it is not surprising that WD will hurt performance of your neural network at some point. 

Let the prediction loss of your net is $\mathcal{L}$ and the weight decay loss $\mathcal{R}$. 

Given a coefficient $\lambda$ that establishes a tradeoff between the two, one optimises 

$$\mathcal{L} + \lambda \mathcal{R}.$$

At the optimium of this loss, the gradients of both terms will have to sum up to zero:

$$ \triangledown \mathcal{L} = -\lambda \triangledown \mathcal{R}. $$

This makes clear that we will not be at an optimium of the training loss. Even more so, the higher $\lambda$ the steeper the gradient of $\mathcal{L}$, which in the case of convex loss functions implies a higher distance from the optimum.

Resources:
[1](https://arxiv.org/pdf/1802.07042.pdf), 
[2](https://stats.stackexchange.com/a/117625/228453)