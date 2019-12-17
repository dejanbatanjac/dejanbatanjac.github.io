---
published: false
layout: post
title: Transformers
permalink: /transformers/
---

## What about transformers?

Transformers (before: pytorch-transformers) is a set of different models (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, CTRL...) for different NLP tasks. 

NLP tasks are (just to recall): 
* Common Sense Reasoning 
* Question Answering
* Cross-Lingual Natural Language Inference 
* Named Entity Recognition 
* Speech Tagging 
* Speech Recognition
* Topic Modeling
* Language Modeling
* Translation
* Text to Speech and Speech to Text Conversion
* Sentiment Classification
* Spell Checking
* Common Sense Reasoning ...

[Transformers repo](https://github.com/huggingface/transformers) has: 10+ architectures with over 30+ trained models, in more than 100 languages.

It supports both TensorFlow 2.0 and PyTorch.

At February 14th 2019 [openai](openai.com) released the article [Better Language Models
and Their Implications](https://openai.com/blog/better-language-models/) where they said:

>Our model, called GPT-2 (a successor to GPT), was trained simply to predict the next word in 40GB of Internet text. Due to our concerns about malicious applications of the technology, we are not releasing the trained model. As an experiment in responsible disclosure, we are instead releasing a much smaller model for researchers to experiment with, as well as a technical paper.

[Transformers repo](https://github.com/huggingface/transformers) idea is to have openai and similar models public. 

## Models you will find inside transformers:
Here is the list of the models in question: 


* BERT (from Google) released with the paper BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

* GPT (from OpenAI) released with the paper Improving Language Understanding by Generative Pre-Training by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.

* GPT-2 (from OpenAI) released with the paper Language Models are Unsupervised Multitask Learners by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.

* Transformer-XL (from Google/CMU) released with the paper Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.

* XLNet (from Google/CMU) released with the paper ​XLNet: Generalized Autoregressive Pretraining for Language Understanding by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.

* XLM (from Facebook) released together with the paper Cross-lingual Language Model Pretraining by Guillaume Lample and Alexis Conneau.

* RoBERTa (from Facebook), released together with the paper a Robustly Optimized BERT Pretraining Approach by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.

* DistilBERT (from HuggingFace), released together with the paper DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter by Victor Sanh, Lysandre Debut and Thomas Wolf. The same method has been applied to compress GPT2 into DistilGPT2, RoBERTa into DistilRoBERTa, Multilingual BERT into DistilmBERT and a German version of DistilBERT.

* CTRL (from Salesforce) released with the paper CTRL: A Conditional Transformer Language Model for Controllable Generation by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher.

* CamemBERT (from Inria/Facebook/Sorbonne) released with the paper CamemBERT: a Tasty French Language Model by Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.

* ALBERT (from Google Research and the Toyota Technological Institute at Chicago) released with the paper ALBERT: A Lite BERT for Self-supervised Learning of Language Representations, by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.


* T5 (from Google AI) released with the paper Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer by Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu.

## Tracking the SOTA

You can track the SOTA for the models. For example:

You can check [XLNet](https://paperswithcode.com/paper/xlnet-generalized-autoregressive-pretraining) achieved SOTA on 20+ NLP tasks.

## Using the models

You actually need to search the `.bin` files on https://huggingface.co/ website. This will give you all the list of all models available:

`site:https://huggingface.co/ "bin"`