---
published: false
layout: post
title: BERT predicts the words
permalink: /bert-word-predicting/
---

In 🤗 there are different BERT classes:

* BertConfig  
* BertTokenizer 
* BertModel 
* BertForPreTraining 
* BertForMaskedLM 
* BertForNextSentencePrediction 
* BertForSequenceClassification 
* BertForMultipleChoice 
* BertForTokenClassification 
* BertForQuestionAnswering 

Possible more will be added soon.
BERT has been trained on the Toronto Book Corpus and Wikipedia.

No need for labeled dataset
pattens in language extraction

attention heads.

Language modeling task...
multiple masked tokens
single masked tokens

calculate probabilities of words.
small number of missing words
just a single missing word

[PAD] in Bert

Bert encoder vs. Bert decoder
probability of sentence
What comes next.
contextual word embeddings.
downstream tasks.


The training was semi-supervised and based on two tasks:
BERT is trained on a masked language model-ingobjective.
* masked language modeling
* next sentence prediction on a large textual corpus

After the training process BERT models were able to understands the language patterns such as grammar.

Let we in here just demonstrate `BertForMaskedLM` simple making a word and predicting words with high probability from the BERT dictionary. 

```
!pip install transformers
from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

sentence = f"McDonald's creates [MASK] food."
token_ids = tokenizer.encode(sentence, return_tensors='pt')
print(token_ids)


```