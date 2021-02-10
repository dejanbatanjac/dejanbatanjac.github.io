---
published: true
layout: post
title: Random Forest
---
- [Intro](#intro)
- [Do we need a validation dataset dealing with RF?](#do-we-need-a-validation-dataset-dealing-with-rf)
- [Where to start with RF?](#where-to-start-with-rf)
- [Inside RF | How Random Forest work](#inside-rf--how-random-forest-work)
- [Extra Trees](#extra-trees)

## Intro

Random Forest (RF) is universal machine learning technique.
It's a way of predicting category or continuous variables with columns of any kind that we first convert to numbers (floats).

Random forests take all kind of data from from categories and continuos values even including: 

* pixel data (images)
* zip codes 
* random data
* labels
* names
* dates ...

The data often requires modification in order to fit into RF, but often just a minor modification (data engineering).

Most of the data engineering will be:

* take the log of some column as input instead of original data column
* convert your date columns into multiple convenient columns such as year, month, day, day of the week...
* convert categories (classes) to numbers 
* convert names to numbers ...

After preparing the data you will provide all the independent variables, and the dependent variable as a parameter to fit the RF.

The independent variables are multiple columns also called <strong>input samples</strong>, and the dependent variable is a single column or <strong>target</strong>. 

Simplified at the end RF will provide the score.

It may be the $R^2$ score also known as [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) for regression problems.

$R^2$ is the proportion of the variance in the dependent variable that is predictable from the independent variable(s).

$R^2$ can take values from -inf, to 1, where 1 is excellent, while 0 and anything less than 0 is considered very bad.

For RF and classification problems the score is the **mean accuracy** on the given test data.

>What is good when dealing with RF:

* RF doesn't assume your data distribution.
* RF doesn't assume the data relationship. 
* RF doesn't assume data interactions.
* RF is very hard to overfit


## Do we need a validation dataset dealing with RF?

It is best when we have a separate validation set, but we can get away even if we don't. We may use part of the test dataset as the validation dataset, and this is unique to random forests. This is called Out-Of-Bag prediction (**OOB**).

OOB is based on the fact that we don't take all the rows (observations) when creating the tree. Instead we may take just 63%, and the remaining 27% observations may be used for the validation.

> In scikit-learn to create the tree for OOB, we pass **oob_score=True**, and then we will have the **oob_score_** at the end that should have the similar coefficient of determination like we have used the separate validation set.

Usually the validation set will take most recent data.

## Where to start with RF?

Probable one very nice starting point is to use the [scikit](https://scikit-learn.org).

For the regression tasks we can start with [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

For the classification tasks we can start with the [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

One nice thing I noticed with scikit RF, is you can do tasks in parallel (multiple processor support exists).

## Inside RF | How Random Forest work

To understand how RF works, we first need to understand how **Decision Tree** (DT) works.

Decision tree simple uses algorithms like:

* ID3 (Iterative Dichotomiser 3)
* C4.5 (successor of ID3)
* CART (Classification And Regression Tree)
* Chi-square automatic interaction detection (CHAID)
* MARS: extends decision trees to handle numerical data better

At the very core algorithm tries to find the best binary split for the data so the **entropy** or **gini** is minimized in the branches taking in account the number of elements.

You can think of entropy or gini are measures of purity. So the best split intuitively will be on a feature that has the highest correlation to the target. We think of the algorithm like finding that feature randomly or with some heuristic and splits so that the weighted average $n_1*e_1 + n_2*e_2$ has the lowest possible score.

This means we need to eliminate the entropy and at the same time to create spits that are close to even in terms of the number of tree elements.

Random forest are random because:
* each DT is trained and validated on slightly different data
* each DT may take slightly different features 


Final **RF bias** should be like the single DT bias.
**RF variance** decreases when we combine trees and thus we decrease the chances of overfitting.


## Extra Trees

Similar to **random forest** are **extra trees**.

Random forest is named because each tree inside uses random subset of datasets for training and validation. 

> The ratio should be 1/e for validation and 2/e for training so each tree will effectively use the whole data. 

While random forest would use fraction of features to test all possible splits first, extra trees will split without testing. 

This is why extra threes need to be much deeper in oder to work.
