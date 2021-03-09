---
published: false
layout: post
title: Validation Schemes
permalink: /validation
---
- [Train, validation and test sets](#train-validation-and-test-sets)
- [Errors](#errors)
- [Decide on a cross validation strategy](#decide-on-a-cross-validation-strategy)
- [Validation schemes](#validation-schemes)
- [What is cross validation](#what-is-cross-validation)



## Train, validation and test sets

We know the train and validation sets are needed, so we can estimate the our machine learning algorithm progress.

However, there should be a **test set** that we should not use touch until the final model. It is also called a hold out set.

If we use the test set before the final model, it will help us fine tune our model, but this will not be **unbiased evaluation** of a final model.

> The term **validation set** is sometimes used instead of **test set**, but this is not a good practice.

## Errors

Errors due to the bias is called underfitting. Errors due to the variances (overfitting).

![bias variance trade off](/images/2021/02/prediction-error-model-complexity.png)



## Decide on a cross validation strategy 

Random k-fold validation is what usually works if the dataset is shuffled and not stratified.

If the **test set** is in the **future**, than this means the train and validation should be positioned so that the validation is also in the future. This is called **train validation time dependence**.

The other important note on cross validation is the usage of **stratified validation**. 

If we have a dataset representing man and woman with different shopping behavior we should try to get the insight if there are many more man than woman in the test set, or if the number is equal. 

When we project our validation dataset it should mimic the test set.

> Strictly speaking even time dependence is a type of stratified validation.

Another similar case is when you need to pay attention on locations. You need to ensure you divide your data to train and validation sets that will be consistent to the test set.

For instance if the test set has two time more rows in US than in Canada, our validation set should reflect that.




## Validation schemes

This page contains information about main validation strategies (schemes): holdout, K-Fold, LOO.

The main rule you should know — never use data you train on to measure the quality of your model. The trick is to split all your data into training and validation parts. 

Below you will find several ways to validate a model.

a) Holdout scheme:

    Split train data into two parts: partA and partB.
    Fit the model on partA, predict for partB.
    Use predictions for partB for estimating model quality. Find such hyper-parameters, that quality on partB is maximized.

b) K-Fold scheme:

    Split train data into K folds. 
    Iterate though each fold: retrain the model on all folds except current fold, predict for the current fold.
    Use the predictions to calculate quality on each fold. Find such hyper-parameters, that quality on each fold is maximized. You can also estimate mean and variance of the loss. This is very helpful in order to understand significance of improvement.

c) LOO (Leave-One-Out) scheme:

    Iterate over samples: retrain the model on all samples except current sample, predict for the current sample. You will need to retrain the model N times (if N is the number of samples in the dataset).
    In the end you will get LOO predictions for every sample in the trainset and can calculate loss.  

Notice, that these are validation schemes are supposed to be used to estimate quality of the model. When you found the right hyper-parameters and want to get test predictions don't forget to retrain your model using all training data

## What is cross validation

If you ask Wikipedia:

> **Cross-validation** (CV), sometimes called **rotation estimation** or **out-of-sample testing**, is any of various similar model validation techniques for assessing how the results of a statistical analysis will generalize to an independent data set.

In other words CV is a technique to achieve a good generalization for your model. 

> Strictly speaking it is a regularization procedure to eliminate the **overfitting**.