---
published: false
layout: post
title: Remove the last layer in Resnet
---

Resnet models are made for 1000 output classes (fine grained classification problem), meaning the last fully connected layer (fc) has output 1000.

The tail of resnet18 for instance looks like this:



If we would like to have just N classes at output we can do multiple things:

* Create resnet from scratch with N outputs 
        
        r = models.resnet18(num_classes=N, pretrained=False)
* Add another linear layer at the very end with input 1000 and output N: 
* Remove the last fc and replace it with the linear layer with output N
* Replace the last fc with `AdaptiveAvgPool2d(1)` followed by fc 512 in and N output

After that we can train 







