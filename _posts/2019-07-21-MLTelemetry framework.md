---
published: false
layout: post
title: ML Telemetry framework
---
>Telemetry is the process of recording and transmitting the readings of an instrument.

I started to realize tha the most needed feature in machine learning is to have the good telemetry and graphical feedback when examining models.

This is why I came up with an idea to create one where the focus is on telemetry. So I set some todo:

* It should be based on PyTorch 
* Designed for convolution problems at start
* Include MNIST, and CIFAR10 examples
* Should support lazy loading
* It should contain several predefined models based on PyTorch Linear, Conv, BN base (i.e. wrapper around conv2d layer and batch norm)
* For simplicity it should not use WD and dropout
* It should have Batch implementation
* It should have accuracy of the validation set for the epoch, as well as batch accuracy formulas.
* It should implement Adam and SGD
* It should have heat map based on Conv
* It should have detailed model summary including `nelement()`
* It should use PyTorch callback system (hooks)
* It should have training and validation steps well defined
* It should support half precision
* It should have VGG and Resnet, Inception, Deepnet modules
* It should have layer statistics (mean, std, sad)
* It should be based on running statistics, (running mean, etc.)

Possible it should also have:
* Lambda layers
* LR finder
* CycLR 
* Draw pictures of intermediate (for conv)
* Draw pictures of intermediate gradients








