---
published: false
layout: post
title: Training sentiment classification using BERT
permalink: /bert-sentiment-classification/
---


phrase-based translation system (see, e.g., Koehn et al., 2003)

The conventional approach to neural machine translation, called an encoder–decoder approach, encodes a whole input sentence into a fixed-length vector from which a translation will be decoded.



* searching for the sentence that maximizes the conditional probability.


Would you be able to answer the question how syntactic and semantic information is saved (stored) inside BERT?

The novel *edge probing* technique exists created/defined by  Tenney et al (2019) in the paper 

Grading sentences:
* grammatically not correct
* grammatically correct
* grammatically correct but false
* grammatically correct and somewhat true
* grammatically correct and true


<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We spend our time finetuning models on tasks like text classif, NER or question answering.<br><br>Yet 🤗Transformers had no simple way to let users try these fine-tuned models.<br><br>Release 2.3.0 brings Pipelines: thin wrappers around tokenizer + model to ingest/output human-readable data. <a href="https://t.co/ZcPTXOJsuS">pic.twitter.com/ZcPTXOJsuS</a></p>&mdash; Hugging Face (@huggingface) <a href="https://twitter.com/huggingface/status/1208141567137058816?ref_src=twsrc%5Etfw">December 20, 2019</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script> 


## BERT on movie reviews

If you still haven't used [MASS](https://www.microsoft.com/en-us/research/blog/introducing-mass-a-pre-training-method-that-outperforms-bert-and-gpt-in-sequence-to-sequence-language-generation-tasks/], you may be using BERT.

We know BERT is used for:
* single sentence classification task
* sentence pair classification task
* multi-label classification
* question answering task (limited)
* next sentence prediction 
* missing word prediction
* speech tagging (noun, adverb, verb, ..)
* whiter sentence is grammatically correct and more ...

However, BERT is not used for common language modeling tasks such as next word prediction. For that you may use [GPT-2]()

Previous tasks were extracted from the The [BERT paper](https://arxiv.org/abs/1810.04805).
Let's use BERT on or movie review task.



labelrd vs. unla beled loss vs. ougputs of the m9e3o

