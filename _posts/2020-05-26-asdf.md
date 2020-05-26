---
published: false
layout: post
title: Huggingface Tokenizer
permalink: /huggingface-tokenizer
---
## Tokenizer class

Tokenizer may have or may not have the mask token. Mask token \<mask> is necessary for masked language modeling.

We need to add this special \<mask> token to the tokenizer. This and all the other special tokens will have the _early_ index values starting from 0.

We can find the mask token:  **tokenizer.mask_token**.


Let's start from the data itself.

A `DataCollator` is responsible for creating batches (batching) and pre-processing samples of data as requested by the training loop.

It is the abstract class| 

```python
class DataCollator(ABC):
```

It has a purpose to take a list of samples from a Dataset and collate them into a batch. The single method it implements is **collate_batch**.

```python
def collate_batch(self, features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
```
The first non abstract implementation **collate_batch** is inside class:

```python
class DefaultDataCollator(DataCollator):
```

The main logic inside **collate_batch** method takes a batch and assumes that all features in the batch do have the same attributes, for that it just checks the first feature with the index 0:

```python
first = features[0]
```

The main conditional is checking the key **k**, if this is **label** or **label_ids** or non of these:

```python
for k, v in vars(first).items():
            if k not in ("label", "label_ids") ...
```

_Example:_



Next there is the class:

```python
class DataCollatorForLanguageModeling(DataCollator):
```
This class **DataCollatorForLanguageModeling** has **collate_batch** method **_tensorize_batch** method, **mask_tokens** method, supports masked language model with the probability of 0.15.

The key method in here (because of the masked language modeling) is **mask_tokens** taking just the tensor of input.

```python
labels = inputs.clone()
```

What happens with the labels? 80% of the time we replace labelse with the **tokenizer.mask_token** .

10% of the time random word will appear instead of the "<mask>", and 10% (rest) of the time we just don't do anything, meaning no masking, no altering.




