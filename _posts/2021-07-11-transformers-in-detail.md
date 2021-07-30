---
published: false
layout: post
title: Transformers general questions
permalink: /transformers-general
---

**Q: What are two main blocks transformer models are composed of?**

A: Typically transformers models do have two blocks:

* Encoder 
* Decoder

![encoder-decoder](\../images/2021/07/ed.png)

**Q: What is the purpose of encoder?**

Encoder receives the input and builds the representation of it (features).

**Q: What is the purpose of the decoder?**

Decoder typically uses features from the encoder with the **sequential** outputs. Say, we have three words of a translation; we use these three words to produce the forth word.


**Q: Why are encoder-only models good for?**

NER and classification.

**Q: Why are decoder-only models good for?**

Text generation.

**Q: Why are encoder-decoder models good for?**

The Transformer architecture was originally designed for translation, and also it can be used for text summarization.

**Name few architectures of encoder, decoder and encoder-decoder models?**

Encoder: Bert, Electra
Decoder: GPT-2, CTRL
Encoder-Decoder: Bart, T5, Marian






## Is BERT auto-encoder

BERT is not really an auto-encoder, in the sense that the prediction of non-masked words is ignored during training. That's a very important distinction. It also uses an (non-generative) next-sentence prediction which is another form of self-supervision.