---
published: true
layout: post
title: Planet dataset from Kaggle, detect the criterion FastAi uses
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


My guess was that loss should not be inside `ImageList` class but inside some class, probable set with: 

`self.crit` or `self.loss` or `self.loss_fn`

And I found that pattern inside the class `Learner`:
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
Also note, handy things we discovered: 

* we may set `fp16` precision for the learning process.
* default optimization function is `SGD_Momentum(0.9)` . 

<sub>* Note momentum in here is the Nestorov momentum.</sub>

And so, the loss should not be anywhere before creating the learner object which is something we create using the `cnn_learner` class.


```
learn = cnn_learner(data, models.resnet50, metrics=[acc_02, f_score])
```

Once we have the learner object I got the feedback on loss function like this.

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

