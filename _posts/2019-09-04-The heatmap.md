---
published: true
layout: post
title: Heatmaps
---

#### How to create the heatmap like this in PyTorch?

![IMG](/images/heatmap1.png)

I used resenet18 model to create one. Also I needed to provide the input image to detect the heatmap for:

    url = "https://raw.githubusercontent.com/dejanbatanjac/pytorch-learning-101/master/h.jpg"

To analyse our resent18 model we would need to get layer input and output size information for the whole model. I did some of that in [this demo](https://gist.github.com/dejanbatanjac/61329992b21fa0e8e02a1d8a5c38079d).

Before the last fc layer we have the average pooling layer with the input size of read from the forward hook:

    torch.Size([1, 512, 7, 7])

As we can understand we have batch size we will remove, 512 features and 7x7 maps. If we *average* all 512 features (activations) we will end with the single 7x7 image.

This image will show us what *average activation intensities* or the **heatmap**. 

> Note: the 7x7 size depends on the input image size. After passing convolution layers, if the image is bigger we may end to 10x10 or some other size.

#### What we may use heat maps for?

Consider the next two heatmaps detecting the child vs. older woman.

![IMG](/images/heatmap2.png) 

Note how neural network in first case showed the attention were on the face mostly, and in case of the granny, the attention was on the right shoulder and throat area. 

We may assume it would be better for the net also to put some attention over the face of the older woman. This means the training should probable last longer.

Also heatmaps can help us isolate examples that don't work quite well yet based on image segmentation (face segmentation in this case). 

