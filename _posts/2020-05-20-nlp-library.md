---
published: false
layout: post
title: NLP library
permalink: /nlp
---
Imaging having the databases for all the NLP tasks inside a single library. You don't have to imagine this. This is possible.


## Install nlp
There is a huge library called HuggingFace nlp library you can install and import like this:

```python
! pip install nlp
import nlp
```
Make sure logging is also enabled to get the feedback on things being downloaded for you:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Check the **pyarrow** version from python if it is the same as the latest pyarrow you installed, else restart your notebook.

```python
!pip install -U pyarrow
import pyarrow
!pip show pyarrow
pyarrow.__version__ # should be the same
```

## Do stuff

You can enumerate all the datasets and all the metrics:

```python
datasets = nlp.list_datasets()
metrics = nlp.list_metrics()
```



