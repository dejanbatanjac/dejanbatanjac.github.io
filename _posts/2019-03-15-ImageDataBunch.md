---
published: true
layout: post
title:  "ImageDataBunch"
---
Fastai library works with text, tabular data, collaborative filtering (collab) and vision out of the box. 
  
In fact these are the main fastai divisions or modules.

The vision module of the fastai library contains all the necessary functions to define a Dataset and train a model for computer vision tasks. 

It contains four different submodules to reach that goal:

- `vision.image` contains the basic definition of an Image object
- `vision.transform` contains all the transforms we can use for data augmentation
- `vision.data` contains the definition of `ImageDataBunch` and `DataBunch`
- `vision.learner` lets you build models

If you use `ImageDataBunch` class you can put anything that represents an image into it either for: test, train, or valication.

To create `ImageDataBunch` you can use several methods:

- `create_from_ll` -> creates from labeled lists
- `from_csv` -> creates from a csv file
- `from_df` -> creates from DataFrame
- `from_folder` -> creates from imagenet style dataset in `path` with `train`,`valid` and `test` subfolders.
- `from_lists` -> creates from list of `fnames` in `path`.
- `from_name_func` -> create from list of `fnames` in `path` with `label_func`
- `from_name_re` -> uses regular expression to extract the names

For example, to load MNIST `from_folder` use this:
```
from fastai import *
from fastai.vision import *
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
```

To load Oxford PETS dataset you can use this code:

```
from fastai import *
from fastai.vision import *
path = untar_data(URLs.PETS);
path_anno = path/'annotations'
path_img = path/'images'
pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs)
```
We can print out the datasets:
```
print(data.valid_ds)
print("...")
print(data.train_ds)
print("...")
print(data.test_ds)
```
And this will output:
```
LabelList (1478 items)
x: ImageList
Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224)
y: CategoryList
chihuahua,Maine_Coon,samoyed,pomeranian,japanese_chin
Path: /root/.fastai/data/oxford-iiit-pet/images
...
LabelList (5912 items)
x: ImageList
Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224)
y: CategoryList
pomeranian,Maine_Coon,english_cocker_spaniel,Bengal,Sphynx
Path: /root/.fastai/data/oxford-iiit-pet/images
...
None
```
