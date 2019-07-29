---
published: true
layout: post
title: MNIST dataset
---

Will try to set few checkpoints in here:

* Setting the device
* Lazy loading the dataset
* Calculating the accuracy


### Setting the device
```
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
```

### Lazy loading of the dataset:

In the case of MNIST dataset this is not needed but here is how this looks like:

```
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import *
import torchvision
import matplotlib.pyplot as plt

class MyLazyDataset(Dataset):
    '''on demand dataset loader taking dataset and targets as separate'''
    def __init__(self, ds, targets, ind_range, transform=None):
        self.ds = ds
        self.ind_range = ind_range
        self.targets   = targets
        self.transform = transform
        
    def __getitem__(self, index):
        # returns a tuple
        image = self.ds[index]
        
        if self.transform:
            x = image.to(dtype=torch.float32)  # make it float           
            x = x.reshape(1, -1)               # make it flat
            x = x/128                          # normalize a bit
            
        y = self.targets[index].long()
        
        return x, y
    
    def __len__(self):        
        return len(self.ind_range)
    

dl = DataLoader( torchvision.datasets.MNIST('/data/mnist', train=True, download=True))
tensor = dl.dataset.data

training_range = range(0,50000)
tld = MyLazyDataset(tensor, targets, ind_range=training_range, transform=True) #training lazy dataset
x, y  = tld[0]
print(x.shape,y)

print(len(tld))

# create DataLoader
bs=2
train_dl = DataLoader(tld, batch_size=bs, drop_last=False, shuffle=False)

X,Y = next(iter(train_dl))
print(type(X))
print(X.shape)
print(type(Y))
print(Y)
```

### Calculating the accuracy:

The accuracy is something we gradually improve over time.
We can define the accuracy of the training dataset, while training, but
in fact more important is the accuracy on the validation dataset.

We use our model to predict and the accuracy which is the number of correct predictions divided by all predictions.

We regularly deal with:

* accuracy after one epoch 
* accuracy of the single batch

Here is where you may end after training MNIST for just one epoch:
```
Epoch training accuracy 0.6329326923076923
Epoch validation accuracy 0.7436899038461539
Last batch training accuracy 0.8125
Last batch validation accuracy 0.71484375
```

Some default batch sizes when training MINST are 32, 64, or 100 examples.

We are most interested to calculate the accuracy on the validation dataset, because this really has much sense. 

We can think of other measures to evaluate the accuracy, like the average batch accuracy of the validation dataset to get the actually accuracy.

We can use the Accuracy class blueprint for evaluating the accuracy:

```
class Accuracy():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.correct = 0
        self.targets = 0
    
    def accuracy_correct_total(self, input, targs):  
        input = input.argmax(dim=-1) 
        self.correct += (input==targs).sum().item()
        self.targets += targs.size(0)
        
    def get(self):
        return self.correct/self.targets
```

Here is the example how we can calculate the accuracy for a single batch in case of MNIST dataset:
```
def accuracy(input, targs):  
    'Accuracy of a single batch'
    input = input.argmax(-1) # batch of input predictions   
    return (input==targs).float().mean()  
```