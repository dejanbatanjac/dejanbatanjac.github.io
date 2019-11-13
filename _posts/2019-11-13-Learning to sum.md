---
published: true
layout: post
title: Learning the sum operation (regression)
---

Typical regression problem would be to learn the sum (+) operation.

Although this is fairly easy problem, it is very important to provide some tips on regression, and the advantages over the classification.

If you would set the neural net for `sum` operation as a classification problem you, would solve the limited amount of experiments, those you have trained. 

This is not the case with the regression. Since it can solve with some precision anything close to the input values of your training set.

Here is the [demo](https://gist.github.com/dejanbatanjac/81c60e579849c07b8c9e93cf6a9797b5).

As you may understand from the demo training was on a very limited set of examples:

```
(0.2000, 0.0000) -> tensor([0.2000])
(0.4000, 0.5000) -> tensor([0.9000])
(0.0000, 0.5000) -> tensor([0.5000])
```
where the first tuple (numbers we add) were from the list:

`[0.0, 0.1, 0.2, 0.3, 0.4]`.

Still later on we were able to get very accurate result using the addends not present in the training set: `0.1200 + 0.4100` = `0.5295`, but more interesting the sum was never part of the training set.

The next image shows how the training loss decreased over the time with the training batches advanced:

![IMG](/images/sum1.PNG)

What is even more important to note, even if we were out of the training set input range like in case `0.3000 + 0.7000` we we got pretty accurate result `0.9793`.

This shows regression can be used with some extent as an interpolation (approximation) technique.