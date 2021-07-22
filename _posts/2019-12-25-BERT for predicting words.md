---
published: true
layout: post
title: BERT predicts the words
permalink: /bert-word-predicting/
---
- [BERT = MLM and NSP](#bert--mlm-and-nsp)
- [Classes](#classes)
  - [BertModel](#bertmodel)
  - [BertForPreTraining](#bertforpretraining)
  - [BertForMaskedLM](#bertformaskedlm)
  - [BertForNextSentencePrediction](#bertfornextsentenceprediction)
  - [BertForSequenceClassification](#bertforsequenceclassification)
  - [BertForMultipleChoice](#bertformultiplechoice)
  - [BertForTokenClassification](#bertfortokenclassification)
  - [BertForQuestionAnswering](#bertforquestionanswering)
- [Demo](#demo)
 
In 🤗 (HuggingFace - on a mission to solve NLP, one commit at a time) there are interesting BERT model.
 
## BERT = MLM and NSP
 
BERT has been trained on the Toronto Book Corpus and Wikipedia and two specific tasks: MLM and NSP.
 
* masked language modeling (MLM)
* next sentence prediction on a large textual corpus (NSP)
 
After the training process BERT models were able to understand the language patterns such as grammar.
 
MLM should help BERT understand the language **syntax** such as grammar.
 
The NSP task should return the result (probability) if the second sentence is following the first one. This helps BERT understand the **semantics**.
 
 
## Classes
 
I am analyzing in here just the PyTorch classes, but at the same time the conclusions are applicable for classes with the **TF** prefix (TensorFlow).
 
Base classed related to BERT include:
 
* **BertModel**
* **BertForPreTraining**
* **BertForMaskedLM**
* **BertForNextSentencePrediction**
* **BertForSequenceClassification**
* **BertForMultipleChoice**
* **BertForTokenClassification**
* **BertForQuestionAnswering**
 
There are even more helper BERT classes besides one mentioned in the upper list, but these are the top most classes.
 
 
### BertModel
 
**BertModel** bare BERT model with **forward** method.
 
### BertForPreTraining
 
**BertForPreTraining** goes with the two heads, MLM head and NSP head.
 
 
```python
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
```
where
 
```python
class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
```
 
**self.predictions** is MLM (Masked Language Modeling) head is what gives BERT the power to fix the grammar errors, and **self.seq_relationship** is NSP (Next Sentence Prediction); usually referred as the classification head.
 
### BertForMaskedLM
 
**BertForMaskedLM** goes with just a **single** multipurpose classification head on top.
 
```python
class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
```
 
where
 
```python
class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
```
 
where
 
```python
class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
```
 
where 
 
```python
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
```
in essence is a single dense layer.
 
### BertForNextSentencePrediction
 
**BertForNextSentencePrediction** is a modification with just a single linear layer **BertOnlyNSPHead**.
 
```python
class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
```
 
Where the output dimension of **BertOnlyNSPHead** is a linear layer with the output size of 2.
 
```python
class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
```
### BertForSequenceClassification
 
**BertForSequenceClassification** is a special model based on the **BertModel** with the linear layer where you can set **self.num_labels** to the number of classes you predict.
 
```python
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
```
 
 
### BertForMultipleChoice
 
Bert model for RocStories and SWAG tasks.
Model has a multiple choice classification head on top.
 
### BertForTokenClassification
 
Bert Model with a token classification head on top (a linear layer on top of the hidden-states output).
 
Ideal for NER Named-Entity-Recognition tasks. 
 
```python
class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
```
 
### BertForQuestionAnswering 
 
Bert model for SQuAD task. It has a span classification head (qa_outputs) to compute span start/end logits.
 
```python
class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
```
 
<!-- 
 
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
downstream tasks. -->
 
 
 
 
## Demo
 
 
Let us in here just demonstrate `BertForMaskedLM` predicting words with high probability from the BERT dictionary based on a [MASK].
 
```python
!pip install transformers --quiet
from transformers import BertTokenizer, BertForMaskedLM
import torch
 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
 
sentence = f"McDonald's creates [MASK] food."
token_ids = tokenizer.encode(sentence, return_tensors='pt')
print(token_ids)
 
masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()
masked_position
 
# forward
with torch.no_grad():
    output = model(token_ids)
 
print(output[0].size())
last_hidden_state = output[0].squeeze()
print(last_hidden_state.size())
print(masked_position)
mask_hidden_state = last_hidden_state[masked_position]
print(mask_hidden_state.size())
 
idx = torch.topk(mask_hidden_state, k=20, dim=0)[1]
print(idx)
 
for i in idx:
    word = tokenizer.convert_ids_to_tokens(i.item())
    print(word)
```
 
_Output:_
```
torch.Size([1, 9, 30522])
torch.Size([9, 30522])
5
torch.Size([30522])
tensor([ 3435,  7554,  7708,  4840,  7965,  2047,  2204,  3737,  2613, 23566,
         2822, 27141, 18015,  7216,  1996,  2980, 14469,  7273,  7107,  2796])
fast
organic
frozen
fresh
healthy
new
good
quality
real
vegetarian
chinese
canned
junk
comfort
the
hot
authentic
thai
instant
indian
```
 
 
 
 

