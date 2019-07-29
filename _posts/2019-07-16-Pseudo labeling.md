---
published: true
layout: post
title: Pseudo labeling
---

Pseudo labeling is based on the [paper](http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf) from Lee.

> Basically, the proposed network is trained in a supervised fashion with labeled and unlabeled data simultaneously.

But how this can be? Wouldn't the unlabeled data provide some kind of a problem?

For unlabeled data we call them pseudo-labels we pick up the class which has the maximum predicted probability.

This is in effect equivalent to entropy regularization. 

## The Entropy

<b>Entropy</b>, as it relates to machine learning, is a measure of the randomness in the information being processed. Flipping a coin is an example of an action that provides information that is random.

The randomness is greatest when there is no relationship between flipping and the outcome. For a coin that has no affinity for heads or tails, the outcome of any number of tosses is difficult to predict and the entropy will be: $log_2 2=1$.

A maximum entropy is achieved when all events are equally probable, and thus the outcome has highest uncertainty.

Entropy regularization means just that making all events equally probable as possible.

## Training the neural networks

The idea is that to train the neural network, we don't need just the quality (targets), but also the quantity (inputs).

Multiple inputs will eventually make separation between classes, needed in a semi-supervised learning.  

Pseudo labeling as being shown in the paper will outperform conventional methods for semi-supervised learning on the MNIST handwritten digit dataset with small number of labeled data.

You may say pPseudo labeling opened the new era of semi-supervised learning.


<!-- 
Later this paper from J. Hinton also pointed couple details about pseudo labeling.

Hinton said the great results appear when you use 1/3-rd, or 1/4-th of the pseudo labels.

The idea is to mix together:

* regular training batches
* pseudo labeled batches
* validation data batches
 -->



