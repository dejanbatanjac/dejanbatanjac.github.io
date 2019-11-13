---
published: true
layout: post
title: From Thin Air
---

Karpathy wrote [Convnetjs](https://cs.stanford.edu/people/karpathy/convnetjs/) and in one [demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html) showed how to learn an image via regression.

In here the two linear layers are used instead and single ReLU in between so the model looks like this:

```
model = torch.nn.Sequential(
          torch.nn.Linear(90000, 100),          
          torch.nn.ReLU(True),
          torch.nn.Linear(100, 90000)
        )
```

The image was 150x200 pixels big.
![IMG](/images/fromthinair1.PNG)

The whole work is in this [gist](https://gist.github.com/dejanbatanjac/e929dc4f2b1effcb2513ff5e5b37dd72).

As you may see experimenting with the learning rate showed me interesting things:

![IMG](/images/fromthinair2.PNG)
![IMG](/images/fromthinair3.PNG)

The best learning rate would be probable around 1e3, although you may not even try this at first since usually people experiment with the range smaller than 1.

In here the `torch.nn.functional.l1_loss` loss function was used, but other would work as well.

I could use any of the:
```
inp = t.reshape(-1)[None].cuda()
z = torch.zeros(1,90000).cuda()
r = torch.rand(1,90000).cuda()
```
where `t` is tensor of the image to be learned.
This can be interpreted in order to learn I could use any input, including the image to be learned itself.


