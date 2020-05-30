---
published: true
layout: post
title: HuggingFace Config Params Explained
permalink: /huggingface-config
---
- [BERT models](#bert-models)
  - [bert-base-uncased](#bert-base-uncased)
  - [bert-large-uncased](#bert-large-uncased)
  - [bert-base-cased](#bert-base-cased)
  - [bert-large-cased](#bert-large-cased)
  - [bert-base-multilingual-uncased](#bert-base-multilingual-uncased)
  - [bert-base-multilingual-cased](#bert-base-multilingual-cased)
  - [bert-large-uncased-whole-word-masking](#bert-large-uncased-whole-word-masking)
  - [bert-large-cased-whole-word-masking](#bert-large-cased-whole-word-masking)
  - [bert-large-uncased-whole-word-masking-finetuned-squad](#bert-large-uncased-whole-word-masking-finetuned-squad)
  - [bert-large-cased-whole-word-masking-finetuned-squad](#bert-large-cased-whole-word-masking-finetuned-squad)
  - [bert-base-cased-finetuned-mrpc](#bert-base-cased-finetuned-mrpc)
- [RoBERTa models](#roberta-models)
  - [roberta-base](#roberta-base)
  - [roberta-large](#roberta-large)
  - [roberta-large-mnli](#roberta-large-mnli)
  - [distilroberta-base](#distilroberta-base)
  - [roberta-base-openai-detector](#roberta-base-openai-detector)
  - [roberta-large-openai-detector](#roberta-large-openai-detector)
- [ALBERT models](#albert-models)
  - [albert-base-v1](#albert-base-v1)
  - [albert-large-v1](#albert-large-v1)
- [BART](#bart)
  - [bart-large](#bart-large)
- [GPT2](#gpt2)
  - [gpt2](#gpt2-1)
- [T5](#t5)
  - [t5-small](#t5-small)

<style> table{font-size: 9px; color:gray } </style>
There are four major classes inside HuggingFace library:

* Config class
* Dataset class
* Tokenizer class
* Preprocessor class

The main discuss in here are different **Config** class parameters for different HuggingFace models. Configuration can help us understand the inter structure of the HuggingFace models.

We will not consider all the models from the library as there are 200.000+ models.

Some interesting models worth to mention based on variety of config parameters are discussed in here and in particular config params of those models.


## BERT models

Sets the config parameters for famous BERT models. Here is a review that can help us understand the BERT model better.


### bert-base-uncased 



param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  768
"initializer_range"|  0.02
"intermediate_size"|  3072
"layer_norm_eps"|  1e-12
"max_position_embeddings"|  512
"model_type"|  "bert"
"num_attention_heads"|  12
"num_hidden_layers"|  12
"pad_token_id"|  0
"type_vocab_size"|  2
"vocab_size"|  30522


To explain **max_position_embeddings** which is actually a limitation I created the example. You cannot have more than 512 embedded tokens, meaning your input is limited.

_Example:_

```python
from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

input_ids = torch.tensor(tokenizer.encode("Hello "*511, add_special_tokens=True)).unsqueeze(0)  # bs = 1
# print(input_ids)
o1,o2 = model(input_ids)
print(o1.shape, o2.shape)
```
`"Hello "*510` would work, but `"Hello "*511` throws the error:

_Token indices sequence length is longer than the specified maximum sequence length for this model (511 > 512). Running this sequence through the model will result in indexing errors._


### bert-large-uncased 

 

param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  1024
"initializer_range"|  0.02
"intermediate_size"|  4096
"layer_norm_eps"|  1e-12
"max_position_embeddings"|  512
"model_type"|  "bert"
"num_attention_heads"|  16
"num_hidden_layers"|  24
"pad_token_id"|  0
"type_vocab_size"|  2
"vocab_size"|  30522



**bert-large-uncased** has the same **vocab_size** as the **bert-base-uncased**, but **hidden_size** is bigger and equal to 1024. Hidden size by num of attention heads should be 64.

```python
attention_head_size = int(hidden_size / num_attention_heads)
# 64 = 1024 / 16
```

In the previous case of **bert-base-uncased** we had the same **attention_head_size**:

```python
attention_head_size = int(hidden_size / num_attention_heads)
# 64 = 768 / 12
```

A tip on the attention heads. Each head is a capability to learn independent features. For instance head H1 can learn features f11, f12, and f13. Head H2 can learn some other features f21, f22, f23, and so on. 

Ideally the more heads you have the more language features you can learn. However, there are some papers like Lottery ticket trying to neglect that, saying you can remove most of the heads and you will still have a good model. There are some other papers like Albert saying to share the parameters among heads is smart and memory efficient.

### bert-base-cased 

 

param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  768
"initializer_range"|  0.02
"intermediate_size"|  3072
"layer_norm_eps"|  1e-12
"max_position_embeddings"|  512
"model_type"|  "bert"
"num_attention_heads"|  12
"num_hidden_layers"|  12
"pad_token_id"|  0
"type_vocab_size"|  2
"vocab_size"|  28996



**bert-base-cased** is pretty much the same as **bert-base-uncased** except the vocab size is even smaller. The vocab size directly impacts the model size in MB. Bigger **vocab_size** bigger the model in MB. Usually the case is that **cased** models do have bigger **vocab_size** but in here this is not true. 

> Tokens "We" and "we" are considered to be different for the cased model.

### bert-large-cased 

 

param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"directionality"|  "bidi"
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  1024
"initializer_range"|  0.02
"intermediate_size"|  4096
"layer_norm_eps"|  1e-12
"max_position_embeddings"|  512
"model_type"|  "bert"
"num_attention_heads"|  16
"num_hidden_layers"|  24
"pad_token_id"|  0
"pooler_fc_size"|  768
"pooler_num_attention_heads"|  12
"pooler_num_fc_layers"|  3
"pooler_size_per_head"|  128
"pooler_type"|  "first_token_transform"
"type_vocab_size"|  2
"vocab_size"|  28996


Again the major difference between the base vs. large models is the **hidden_size** 768 vs. 1024, and **intermediate_size** is 3072 vs. 4096.

BERT has 2 x FFNN inside each encoder layer, for each layer, for each position (**max_position_embeddings**), for every head, and the size of first FFNN is:
(**intermediate_size** X **hidden_size**). This is the hidden layer also called intermediate layer.

There is a second FFNN of size (**hidden_size** X **intermediate_size**). This is the output layer.

Two thirds of all BERT parameters goes to the FFNNs.

On the other side **bert-large-cased** is very similar to **bert-large-uncased**, but it has the smaller vocab_size. I think the main reason for smaller vocab size is memory (takes less MB).


### bert-base-multilingual-uncased

 

param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"directionality"|  "bidi"
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  768
"initializer_range"|  0.02
"intermediate_size"|  3072
"layer_norm_eps"|  1e-12
"max_position_embeddings"|  512
"model_type"|  "bert"
"num_attention_heads"|  12
"num_hidden_layers"|  12
"pad_token_id"|  0
"pooler_fc_size"|  768
"pooler_num_attention_heads"|  12
"pooler_num_fc_layers"|  3
"pooler_size_per_head"|  128
"pooler_type"|  "first_token_transform"
"type_vocab_size"|  2
"vocab_size"|  105879



Now we have three times bigger vocab size with **bert-base-multilingual-uncased** compared to **bert-large-cased**. This seams to be a good choice since the model covers 100+ languages.


### bert-base-multilingual-cased


param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"directionality"|  "bidi"
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  768
"initializer_range"|  0.02
"intermediate_size"|  3072
"layer_norm_eps"|  1e-12
"max_position_embeddings"|  512
"model_type"|  "bert"
"num_attention_heads"|  12
"num_hidden_layers"|  12
"pad_token_id"|  0
"pooler_fc_size"|  768
"pooler_num_attention_heads"|  12
"pooler_num_fc_layers"|  3
"pooler_size_per_head"|  128
"pooler_type"|  "first_token_transform"
"type_vocab_size"|  2
"vocab_size"|  119547



This is a very big model it has bigger **vocab_size** compared to  **bert-base-multilingual-uncased**.


### bert-large-uncased-whole-word-masking

 

param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  1024
"initializer_range"|  0.02
"intermediate_size"|  4096
"layer_norm_eps"|  1e-12
"max_position_embeddings"|  512
"model_type"|  "bert"
"num_attention_heads"|  16
"num_hidden_layers"|  24
"pad_token_id"|  0
"type_vocab_size"|  2
"vocab_size"|  30522






### bert-large-cased-whole-word-masking

 

param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"directionality"|  "bidi"
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  1024
"initializer_range"|  0.02
"intermediate_size"|  4096
"layer_norm_eps"|  1e-12
"max_position_embeddings"|  512
"model_type"|  "bert"
"num_attention_heads"|  16
"num_hidden_layers"|  24
"pad_token_id"|  0
"pooler_fc_size"|  768
"pooler_num_attention_heads"|  12
"pooler_num_fc_layers"|  3
"pooler_size_per_head"|  128
"pooler_type"|  "first_token_transform"
"type_vocab_size"|  2
"vocab_size"|  28996




### bert-large-uncased-whole-word-masking-finetuned-squad

Whenever we see finetuned-squad this means this model is prepared for question answering tasks.

param | value
---------|---------- 
"attention_probs_dropout_prob"|  0.1
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  1024
"initializer_range"|  0.02
"intermediate_size"|  4096
"layer_norm_eps"|  1e-12,
"max_position_embeddings"|  512
"model_type"|  "bert"
"num_attention_heads"|  16
"num_hidden_layers"|  24
"pad_token_id"|  0
"type_vocab_size"|  2
"vocab_size"|  30522



### bert-large-cased-whole-word-masking-finetuned-squad

Model has been finetuned on SQUAD. The BERT has been trained on MLM and NSP tasks. These training activities should help BERT learn the grammar and semantics respectively. The two training tasks used different heads, and after the original training, the BERT has been fine tuned on SQUAD. This third task should be the fastest.
 

param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"directionality"|  "bidi"
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  1024
"initializer_range"|  0.02
"intermediate_size"|  4096
"layer_norm_eps"|  1e-12
"max_position_embeddings"|  512
"model_type"|  "bert"
"num_attention_heads"|  16
"num_hidden_layers"|  24
"pad_token_id"|  0
"pooler_fc_size"|  768
"pooler_num_attention_heads"|  12
"pooler_num_fc_layers"|  3
"pooler_size_per_head"|  128
"pooler_type"|  "first_token_transform"
"type_vocab_size"|  2
"vocab_size"|  28996



### bert-base-cased-finetuned-mrpc

MRPC is a mark this model can be used for sequence classification.

 

param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  768
"initializer_range"|  0.02
"intermediate_size"|  3072
"layer_norm_eps"|  1e-12
"max_position_embeddings"|  512
"model_type"|  "bert"
"num_attention_heads"|  12
"num_hidden_layers"|  12
"pad_token_id"|  0
"type_vocab_size"|  2
"vocab_size"|  28996



## RoBERTa models

RoBERTa is modified BERT model trained on 10 times more text. RoBERTa is very similar to BERT.

### roberta-base
 

param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"bos_token_id"|  0
"eos_token_id"|  2
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  768
"initializer_range"|  0.02
"intermediate_size"|  3072
"layer_norm_eps"|  1e-05
"max_position_embeddings"|  514
"model_type"|  "roberta"
"num_attention_heads"|  12
"num_hidden_layers"|  12
"pad_token_id"|  1
"type_vocab_size"|  1
"vocab_size"|  50265



Here is one neat trick to explain what does it actually mean the **hidden_size** and why it is used for. We are using the **roberta-base**.


```python
import torch
from transformers import RobertaModel, RobertaTokenizer
model = RobertaModel.from_pretrained("roberta-base")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
input_ids = torch.tensor(tokenizer.encode("Hello ", add_special_tokens=True)).unsqueeze(0)  # bs = 1
print(input_ids) # tensor([[    0, 20920,     2]])
o1,o2 = model(input_ids)
o1.shape, o2.shape
(torch.Size([1, 3, 768]), torch.Size([1, 768]))
```
In here the **hidden_size** is 768, as config param. Also **bos_token_id** and **eos_token_id** are actually present inside the config file.

Before passing it to RobertaModel we prepare **input_ids** with the **add_special_tokens=True** the input and we add **bos_token** and **eos_token** at the beginning and at the end in that order.

```python
print(input_ids) # tensor([[    0, 20920,     2]])
```

We have \<s>Hello\</s> because RoBERTa uses \<s> and \</s> special tokens. 

> BERT uses [CLS] and [SEP] as starting token and separator token respectively that correspond to RoBERTa tokens we mentioned. Note that RoBERTa also can have a separate classification token, but it is usually equivalent to the bos token (\<s>).

Now if you give above sentence to **RobertaModel** you will get two 768 dimension embeddings for each token in the given sentence.

The _sequence output_ will have dimension [1, 3, 768] since there are 3 tokens including [BOS] and [EOS]. This is the last hidden state.

There is also the **pooled output** ( [1, 1, 768] ) which is the embedding of [BOS] token.

* use **pooled output** for sentence classification.
* use **sequence output** for detecting text similarity for instance.



### roberta-large

 

param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"bos_token_id"|  0
"eos_token_id"|  2
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  1024
"initializer_range"|  0.02
"intermediate_size"|  4096
"layer_norm_eps"|  1e-05
"max_position_embeddings"|  514
"model_type"|  "roberta"
"num_attention_heads"|  16
"num_hidden_layers"|  24
"pad_token_id"|  1
"type_vocab_size"|  1
"vocab_size"|  50265



In here note the **vocab_size** for RoBERTa (**roberta-base** and **roberta-large**) is ~ 50K while for BERT is ~ 30K. Of course, it depends on a model, different models can have arbitrary vocab sizes.


### roberta-large-mnli 

This model is finetuned for sequence classification. See. **RobertaForSequenceClassification**.

**MNLI** means Multi-Genre Natural Language Inference Corpus. This is one of nine GLUE tasks. 

The other GLUE tasks are:

* CoLAThe Corpus of Linguistic Acceptability (ss)
* SST-2The Stanford Sentiment Treebank (ss)
* MRPCThe Microsoft Research Paraphrase Corpus (sim)
* QQPThe Quora Question Pairs (sim)
* STS-BThe Semantic Textual Similarity Benchmark (sim)
* MNLIThe Multi-Genre Natural Language Inference Corpus (inf)
* QNLIThe Stanford Question Answering Dataset (inf)
* RTEThe Recognizing Textual Entailment (inf)
* WNLIThe Winograd Schema Challenge (inf)

> ss: single sentence tasks, sim: similarity tasks, inf: inference tasks

 

param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"bos_token_id"|  0
"eos_token_id"|  2
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  1024
"initializer_range"|  0.02
"intermediate_size"|  4096
"layer_norm_eps"|  1e-05
"max_position_embeddings"|  514
"model_type"|  "roberta"
"num_attention_heads"|  16
"num_hidden_layers"|  24
"pad_token_id"|  1
"type_vocab_size"|  1
"vocab_size"|  50265




### distilroberta-base

Distilled models are using some tricks to downsize the number of parameters and at the same time keep the original model quality the best they can.

Even though some accuracy will be lost, the model size will be 3x smaller.

This particular model is used for masked language modeling (predicting the missing word) that may fix the grammar errors for instance.


param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"bos_token_id"|  0
"eos_token_id"|  2
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  768
"initializer_range"|  0.02
"intermediate_size"|  3072
"layer_norm_eps"|  1e-05
"max_position_embeddings"|  514
"model_type"|  "roberta"
"num_attention_heads"|  12
"num_hidden_layers"|  6
"pad_token_id"|  1
"type_vocab_size"|  1
"vocab_size"|  50265



### roberta-base-openai-detector

The following two models are used for sequence classification **RobertaForSequenceClassification**

 

param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"bos_token_id"|  0
"eos_token_id"|  2
"hidden_act"|  "gelu"
"hidden_dropout_pro"|  0.1
"hidden_size"|  768
"initializer_range"|  0.02
"intermediate_size"|  3072
"layer_norm_eps"|  1e-05
"max_position_embeddings"|  514
"model_type"|  "roberta"
"num_attention_heads"|  12
"num_hidden_layers"|  12
"output_past"|  true
"pad_token_id"|  1
"type_vocab_size"|  1
"vocab_size"|  50265




### roberta-large-openai-detector

As you can see RoBERTa has almost all the same parameters are BERT. 

param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"bos_token_id"|  0
"eos_token_id"|  2
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  1024
"initializer_range"|  0.02
"intermediate_size"|  4096
"layer_norm_eps"|  1e-05
"max_position_embeddings"|  514
"model_type"|  "roberta"
"num_attention_heads"|  16
"num_hidden_layers"|  24
"output_past"|  true
"pad_token_id"|  1
"type_vocab_size"|  1
"vocab_size"|  50265




## ALBERT models

### albert-base-v1

ALBERT is A Lite BERT! Project by Google and Toyota. It brings the new param **num_hidden_groups** that is set to 1. 

With **num_hidden_groups** equal to number of heads we will have BERT again.

 

param | value
---------|----------
attention_probs_dropout_prob| 0.1
bos_token_id| 2
classifier_dropout_prob| 0.1
down_scale_factor| 1
embedding_size| 128
eos_token_id| 3
gap_size| 0
hidden_act| "gelu"
hidden_dropout_prob| 0.1
hidden_size| 768
initializer_range| 0.02
inner_group_num| 1
intermediate_size| 3072
layer_norm_eps| 1e-12
max_position_embeddings| 512
model_type| "albert"
net_structure_type| 0
num_attention_heads| 12
num_hidden_groups| 1
num_hidden_layers| 12
num_memory_blocks| 0
pad_token_id| 0
type_vocab_size| 2
vocab_size| 30000



### albert-large-v1

This model will need just ~70MB to load even though it has **num_attention_heads=16**. Compared to BERT this is 10x less memory. Specifically applicable for handhold devices, cars and household devices.

```python
import torch
from transformers import AlbertModel, AlbertTokenizer
model = AlbertModel.from_pretrained("albert-large-v1")
tokenizer = AlbertTokenizer.from_pretrained('albert-large-v1')
input_ids = torch.tensor(tokenizer.encode("Hello ", add_special_tokens=False)).unsqueeze(0)  # bs = 1
print(input_ids)
```

 

param | value
---------|----------
"attention_probs_dropout_prob"|  0.1
"bos_token_id"|  2
"classifier_dropout_prob"|  0.1
"down_scale_factor"|  1
"embedding_size"|  128
"eos_token_id"|  3
"gap_size"|  0
"hidden_act"|  "gelu"
"hidden_dropout_prob"|  0.1
"hidden_size"|  1024,
"initializer_range"|  0.02
"inner_group_num"|  1
"intermediate_size"|  4096
"layer_norm_eps"|  1e-12
"max_position_embeddings"|  512
"model_type"|  "albert",
"net_structure_type"|  0,
"num_attention_heads"|  16,
"num_hidden_groups"|  1,
"num_hidden_layers"|  24
"num_memory_blocks"|  0
"pad_token_id"|  0,
"type_vocab_size"|  2
"vocab_size"|  30000



Again the major difference between the base vs. large models is the **hidden_size** 768 vs. 1024, and **intermediate_size** is 3072 vs. 4096.



## BART

### bart-large

Applicable for both **BartForMaskedLM** and **BartForSequenceClassification** tasks.

BART is combining the power of BERT and GPT.
To train BART several tricks are used:

* MLM (Masking LM) like BERT, random tokens are sampled and replaced with [MASK] token.
* TD (Token Deletion) where random tokens are deleted from the input. The model then must decide which positions are missing inputs.
* TI (Text Infilling) where several text spans are sampled and replaced with a single [MASK] token (can be 0 lengths).
* SP (Sentence Permutation) where a document is divided into sentences based on full stops. Sentences are shuffled in random order.
* DR (Document Rotation) where token is chosen uniformly at random, and the document is rotated so that it begins with that token and then model has been trained to identify the start of the document.

 

param | value
---------|----------
"_num_labels"|  3
"activation_dropout"|  0.0
"activation_function"|  "gelu"
"add_final_layer_norm"|  false
"attention_dropout"|  0.0
"bos_token_id"|  0
"classif_dropout"|  0.0
"d_model"|  1024
"decoder_attention_heads"|  16
"decoder_ffn_dim"|  4096
"decoder_layerdrop"|  0.0
"decoder_layers"|  12
"decoder_start_token_id"|  2
"dropout"|  0.1
"encoder_attention_heads"|  16
"encoder_ffn_dim"|  4096
"encoder_layerdrop"|  0.0
"encoder_layers"|  12
"eos_token_id"|  2
"init_std"|  0.02
"is_encoder_decoder"|  true
"max_position_embeddings"|  1024
"model_type"|  "bart"
"normalize_before"|  false
"num_hidden_layers"|  12
"output_past"|  false
"pad_token_id"|  1
"prefix"|  " "
"scale_embedding"|  false
"vocab_size"|  50265



## GPT2

There are more powerful gpt2 models but this one is the smallest.

### gpt2

 

param | value
---------|----------
"activation_function"|  "gelu_new"
"attn_pdrop"|  0.1
"bos_token_id"|  50256
"embd_pdrop"|  0.1
"eos_token_id"|  50256
"initializer_range"|  0.02
"layer_norm_epsilon"|  1e-05
"model_type"|  "gpt2"
"n_ctx"|  1024
"n_embd"|  768
"n_head"|  12
"n_layer"|  12
"n_positions"|  1024
"resid_pdrop"|  0.1
"summary_activation"|  null
"summary_first_dropout"|  0.1
"summary_proj_to_labels"|  true
"summary_type"|  "cls_index"
"summary_use_proj"|  true
"vocab_size"|  50257



The meaning of the most important params are:

* **n_positions** e.g., 512 or 1024 or 2048 is what correspond to BERT **max_position_embeddings**.

* **n_ctx** dimension of the causal mask (usually same as **n_positions**)
* **n_embd** dim of embeddings and hidden state (BERT have these but these have different values, while GPT-2 values are the same).
* **n_layer** number of hidden layers in the Transformer encoder.
* **n_head** number of heads


## T5

Used for several tasks (multitask model)


### t5-small

 

param | value
---------|----------
"d_ff"|  2048
"d_kv"|  64
"d_model"|  512
"decoder_start_token_id"|  0
"dropout_rate"|  0.1
"eos_token_id"|  1
"initializer_factor"|  1.0
"is_encoder_decoder"|  true
"layer_norm_epsilon"|  1e-06
"model_type"|  "t5"
"n_positions"|  512
"num_heads"|  8
"num_layers"|  6
"output_past"|  true
"pad_token_id"|  0
"relative_attention_num_buckets"|  32
"vocab_size"|  32128




* **d_model** is size of the encoder layers and the pooler layer it is the same was **hidden_size** in BERT
* **d_ff** is hidden layer size of the FNN (Feed Forwarded Network)
* **d_kv** is self.hidden_size // self.num_attention_heads, the same as **attention_head_size** in BERT

