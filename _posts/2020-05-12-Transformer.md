---
published: false
layout: post
title: NLP Transformer model
permalink: /transformer-model/
---
## NLP in the early days

Before the Transformer, NLP state of the art models used the bag-of-words approach, but this approach was missing the order of the words or positional encoding.

With N-grams bag-of-words technique evolved but still it was just an improvement of the **statistical approach**.

With the idea of RNN NLP started to use **deep learning** as an improvement to the statistical approach to NLP. 

The problem with the RNN technique in the early days, was gradient explosion/vanishing that was kind fixed with the introduction of LSTM.

Still LSTM was inefficient to parallelize operation due to _sequential computation_ that inhibits parallelization.


## The Transformer

In 2017 Ashish Vaswani and friends developed the [**Transformer** model](https://arxiv.org/abs/1706.03762)}{:rel="nofollow"} in a paper _Attention os all you need_.

If I would have just two words to describe the Transformer I would use:

* self-attention
* multi-headed

> actually these are 4 words ;)

## The Attention

The original idea of attention is to create the weighted relation between the input and the output.

![attention](/images/attention.jpg) 

_An image from [Chris Olah](https://distill.pub/2016/augmented-rnns/){:rel="nofollow"}_

A. Vaswani got the idea to use self-attention for words representation.

With Transformer the **self-attention** idea is to create the average of encoded neighbor words.

The self-attention is just a weighted average of all the encoded neighbor words.

If $X$ is a matrix of encoded words where each row is a single word, and the number of columns is the length of each word then after the self attention we get the matrix $Y$ with the same dimension as $X$.

## The Q-K-V matrices

Self attention can be expressed as:

$$QK^T$$

We use the fact that in a sentence each **pair of words** are connected. We can use math to formalize these connections. 

For each word that we select (**query word**) we will query all the other **key words** including the query word itself.

If we have $n$ words, then we will have $n^2$ connections or $n^2$ vector products. 

All the queries and keys we will learn as the embeddings matrices $Q , K$.

Additional $V$ matrix we add as FFN (Feed Forward linear network layer). This matrix is the embedding matrix we use to extract the features (semantical juice).


The $Q, K, V$ matrices called Query, Key, are used together and will form the Z matrix:

$Z = softmax(QK^T)V$

The product $QK^T$ can also be normalized and additionally we can modulate (add) the relative attention function: $QK^T+Qf_{rel}$


We pass the product to the **softmax** function to get the results in between 0 to 1 that all sum to 1.

After the softmax part we will dot product the result with the $V$. $V$ will be used to extract the hidden features.


## Multiple heads

By concatenating all the $Z$ results for all teh heads we get the final $Z_*$ matrix.

![transformer architecture](/images/transformer.jpg)

> The Nx part meaning is for each N heads.


You can think a single head is like a decision tree, and a transformer model is like a random forest.

Each head learns specific language features and improves while training. Each head can typically learn features like: Who (the subject), Did what (the action), to whom (the object), where (location) and so on.

The more heads we have the better.

## BERT

One of the favorite transformer models is called BERT. Here is how the head of the basic BERT model looks like in PyTorch:

```
(transformer): Transformer(
    (layer): ModuleList(
      (0): TransformerBlock(
        (attention): MultiHeadSelfAttention(
          (dropout): Dropout(p=0.1, inplace=False)
          (q_lin): Linear(in_features=768, out_features=768, bias=True)
          (k_lin): Linear(in_features=768, out_features=768, bias=True)
          (v_lin): Linear(in_features=768, out_features=768, bias=True)
          (out_lin): Linear(in_features=768, out_features=768, bias=True)
        )
        (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (ffn): FFN(
          (dropout): Dropout(p=0.1, inplace=False)
          (lin1): Linear(in_features=768, out_features=3072, bias=True)
          (lin2): Linear(in_features=3072, out_features=768, bias=True)
        )
        (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      )
```
There is certain amount of memory needed for each head. The total memory for the upper TransformerBlock is:
```
3x716x716=1.537.968 (~1,5MB) 
```
The FFN (Feed Forward Network) block size is:

```
3072x716+716x716=2.712.208 (~2.7MB)
```

## The illustrated transformer

The [illustrated transformer](http://jalammar.github.io/illustrated-transformer/){:rel="nofollow"} is an illustrated helper to understand the transformer model.

## Why Transformers may be used for?

The Transformer is unsupervised language model that learns from both inputs and outputs. It can be used for:

* masked word prediction
* next sentence prediction
* machine translation
* sequence classification
* multiple choice selection
* token classification
* question answering


## Skip connections and residuals

## Using PyTorch or TensorFlow

```python
import tensorflow as tf
print(tf.__version__)
import torch
print(torch.__version__)
```

## Appendix: NLP tasks

![tasks in NLP](/images/tasks.jpg)

Transformer architecture shows nice results in Machine Translation (BLEU): EN->DE 28.4 and 41.8 for EN->FR according to the original paper.


```python
import torch
from pytorch_transformers import *

# Simple and standard API for 6 transformer architectures & 27 pretrained model weights:

MODELS = [(BertModel, RertTokenizer 'bert-base-uncased'),
(OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
(GPT2Model, GPT2Tokentizer, 'gpt2'),
(TransfoXLModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
(XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
(XLMModel, XLMTokentizer, 'xlm-mlm-enfr-1024')]

# Let's encode some text in a sequence of hidden-states using each model:
for model_class, tokenizer_class, pretrained_weights in MODELS:
# Load pretrained model/tokenizer
tokenizer=tokenizer_class.from_pretrained(pretrained_weights)
model=model_class.from_pretrained(pretrained_weights)

# Encode text
input_ids=torch.tensor([tokenizer.encode("Here is some text to encode")])
last_hidden_states=model(input_ids)[0]  # Models outputs are now tuples

# Models can return full list of hidden-states & attentions weights at each layer
model=model_class.from_pretrained( pretrained_weights,
                                  output_hidden_states=True, 
                                  output_attentions=True)
input_ids=torch.tensor([tokenizer.encode("Let's see hidden-states and attentions")])

all_hidden_states, all_attentions=model(input_ids)[-2:]

# Models are compatible with Torchscript
model=model_class.from_pretrained(pretrained_weights, torchscript=True)
traced_model=torch.jit.trace(model, (input_ids,))

# Simple serialization for models and tokenizers
model.save_pretrained('./directory/to/save/')  # save
model=model_class.from_pretrained('./directory/to/save/')  # re-load
# SOTA examples for GLUE, SQUAD, text generation...
```
