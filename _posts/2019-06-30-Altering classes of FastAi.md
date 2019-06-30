---
published: true
layout: post
title: Altering classes of FastAi
---
Wanted to show one interesting feature of FastAI library that I though at first is inconsistency. 

It addresses creating, splitting, and labeling data.

The following code will import the `PLANET_TINY` dataset and print out the classes in use after every step.

```
import fastai
from fastai.vision import *
path_data = untar_data(URLs.PLANET_TINY); path_data.ls()

def bases(obj):    
    'Will provide obj class and base classes'
    print("...")
    print(type(obj))
    print(obj.__class__.__name__)
    for base in obj.__class__.__bases__:
        print(base)
        print (base.__name__)

        
# C:/Users/dj/.fastai/data/planet_tiny/train/labels.csv
c = ImageList.from_csv(path_data, 'labels.csv', folder='train-jpg', suffix='.jpg')
bases(c)   
c = c.split_by_rand_pct(0.2)
bases(c)   
c.label_from_df(label_delim=' ')
bases(c)  
```
Out
```
...
<class 'fastai.vision.data.ImageList'>
ImageList
<class 'fastai.data_block.ItemList'>
ItemList
...
<class 'fastai.data_block.ItemLists'>
ItemLists
<class 'object'>
object
...
<class 'fastai.data_block.LabelLists'>
LabelLists
<class 'fastai.data_block.ItemLists'>
ItemLists
```
As you can see, we first create `ImageList` (base in ItemList), and then after the splitting we get `ItemLists`, and after the labeling we get `LabelLists` (based in `ItemLists`).

All classes do implement `__call__` so we can write like this as well:
```
c = (ImageList.from_csv(path_data, 'labels.csv', folder='train-jpg', suffix='.jpg')
        .split_by_rand_pct(0.2)
        .label_from_df(label_delim=' '))
```

