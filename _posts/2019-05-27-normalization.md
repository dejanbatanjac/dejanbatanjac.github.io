---
published: true
---
As you may assume in PyTorch images we use should be prepared for learning. This is the image normalization process at the very essence. There are of course the other things we can do but our focus is on normalization.

In case we used PIL `Image` class and converted from PIL Image to PyTorch Tensor.

    from PIL import Image
    from torchvision.transforms import ToTensor
    
Our Tensor images will look like this:

    tensor([[[0.1725, 0.2353, 0.2941,  ..., 1.0000, 1.0000, 0.9843],
             [0.1804, 0.2314, 0.2824,  ..., 1.0000, 1.0000, 0.9843],
             [0.1922, 0.2314, 0.2706,  ..., 1.0000, 1.0000, 0.9843],
             ...,
             [0.7098, 0.6863, 0.6627,  ..., 0.6863, 0.6863, 0.6039],
             [0.6941, 0.6824, 0.6745,  ..., 0.6118, 0.6314, 0.5569],
             [0.6863, 0.6863, 0.6941,  ..., 0.2392, 0.2902, 0.3137]], ...


Note how, `min` and `max` value of this tensor will be: `tensor(0.)` and `tensor(1.)` respectively. 
The histogram per channel will look like this:

...![]({{site.baseurl}}/images/normalization1.png)

Let we use the following PyTorch normalization function:

    def normalize(x: torch.FloatTensor, mean: torch.FloatTensor, std: torch.FloatTensor) -> torch.FloatTensor
        "Normalize `x` with `mean` and `std`."
        return (x - mean[..., None, None]) / std[..., None, None]
        
What we provide is a Tensor image `x` and `mean` and `std` values for the image set we are working in.
This means that we evaluated in advance the mean and std for all the images in the set.

Following are some well known `mean` and `std` list tupples for RGB channels: 

    cifar_stats = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])
    imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    mnist_stats = ([0.15] * 3, [0.15] * 3)
    
In particular `[0.491, 0.482, 0.447]` is the mean for the cifar image set; `0.491` is the mean for the Red channel, and so on. The standard deviation for the same image set is represented with this list `[0.247, 0.243, 0.261]` and `0.247` is exactly the std of the Red channel.

        
Note the Ellipsis notation we used inside `normalize` function. It may be strange what it means in PyTorch.
Let's check this code:

    l=Tensor([1,2,3])
    print(l)
    r=l[...,None, None]
    print(r)

This will output like this:

    tensor([1., 2., 3.])
    tensor([[[1.]],

            [[2.]],

            [[3.]]])

Note how we subtract `x - mean[..., None, None]` for specific RGB channel, and also how we do RGB channel division `std[..., None, None]` after that.

At the end, we will get the result like this setting our data around 0.

...![]({{site.baseurl}}/images/normalization2.png)

You may note that before we had our tensor values between [0., 1.], and now we have positive and negative values around 0, ideal for machine learning.
