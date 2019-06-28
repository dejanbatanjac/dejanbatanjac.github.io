---
published: true
layout: post
title: Planet dataset from Kaggle criterion to use
---
I decided to fast-forward the Planet dataset from Kaggle from [planet competition](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space), or the alternative name of this challenge was "understanding the amazon from space". It touches the problem of wood cutting.

I used [FastAi](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-planet.ipynb) to examine things, specially I wanted to examine what kind of loss function will be used in this case of multi-label classification problem.

There was a general note `ImageList` class should be used for this set:

```
np.random.seed(42)
src = (ImageList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' '))
```

Then I printed something like this `src` object (class):

```
print(type(src))
print(src.__class__.__name__)
for base in src.__class__.__bases__:
    print(base)
    print (base.__name__)
```
Output:
```
<class 'fastai.data_block.LabelLists'>
LabelLists
<class 'fastai.data_block.ItemLists'>
ItemLists
```

OK, what is wrong here, I provided the `ImageList` class and I got the class `LabelList` which has base in `ItemLists`.

Why is that? I expected to get the `ImageList` class.

I made a quick test and noted minor inconsistency that you can dupe with these lines of code:

```
import fastai
from fastai.vision import *
path_data = untar_data(URLs.PLANET_TINY); path_data.ls()

src = ImageList.from_folder(path_data/'train');
# print(src)

print(type(src))
print(src.__class__.__name__)
for base in src.__class__.__bases__:
    print(base)
    print (base.__name__)

# <class 'fastai.vision.data.ImageList'>
# ImageList
# <class 'fastai.data_block.ItemList'>
# ItemList
  

print("...")
# import ipdb
# ipdb.set_trace()

# C:/Users/dj/.fastai/data/planet_tiny/train/labels.csv
src2 = (ImageList.from_csv(path_data, 'labels.csv', folder='train-jpg', suffix='.jpg')
        .split_by_rand_pct(0.2)
        .label_from_df(label_delim=' '))

print(type(src2))
print(src2.__class__.__name__)
for base in src2.__class__.__bases__:
    print(base)
    print (base.__name__)

# <class 'fastai.data_block.LabelLists'>
# LabelLists
# <class 'fastai.data_block.ItemLists'>
# ItemLists
```

Found out that `label_from_df` method actually calls this peace of code 
```
 def _inner(*args, **kwargs):
            self.train = ft(*args, from_item_lists=True, **kwargs)
            assert isinstance(self.train, LabelList)
            kwargs['label_cls'] = self.train.y.__class__
            self.valid = fv(*args, from_item_lists=True, **kwargs)
            self.__class__ = LabelLists
            self.process()
            return self
        return _inner
```
which at the very end sets our class to `LabelLists`, however the inconsistency is still there.

But now when I know what caused this, it is the time to check on the loss function type.

My guess was that loss should be somewhere inside some class, probable set with: 

`self.crit` or `self.loss` or `self.loss_fn`

And I found that inside the class `Learner`:
```
class Learner():
    def __init__(self, data, models, opt_fn=None, tmp_name='tmp', models_name='models', metrics=None, clip=None, crit=None):
        
        self.data_,self.models,self.metrics,self.clip = data,models,metrics,clip
        self.sched=None
        self.wd_sched = None
        self.opt_fn = opt_fn or SGD_Momentum(0.9)
        self.tmp_path = tmp_name if os.path.isabs(tmp_name) else os.path.join(self.data.path, tmp_name)
        self.models_path = models_name if os.path.isabs(models_name) else os.path.join(self.data.path, models_name)
        os.makedirs(self.tmp_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        self.crit = crit if crit else self._get_crit(data) # <---
        self.reg_fn = None
        self.fp16 = False
```
Also note, this is the way how I discovered where we can set `fp16` precision for the learning process. There you can see the default optimization function is `SGD_Momentum(0.9)` *. 

<sub>* Note momentum in here is the Nestorov momentum.</sub>

The loss should not be anywhere before creating the learner object which is something we create using the `cnn_learner` class.

Note that the next few lines demonstrate Python partials - literally functions with some parameters set.

```
acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, models.resnet50, metrics=[acc_02, f_score])
```

Once we have the learner object I got the feedback on loss function.

```
print(learn.loss_func)
print("...")
print(learn.loss_func.func)
```
Out:
```
FlattenedLoss of BCEWithLogitsLoss()
...
BCEWithLogitsLoss()
```

This is nearly what I expected for this kind of problem so I set a little ☑.

