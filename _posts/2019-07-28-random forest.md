---
published: true
layout: post
title: Random Forest
---

Random Forest (RF) is universal machine learning technique.
It's a way of predicting category or continuous variables with columns of any kind that we first convert to numbers (floats).

Random forests take all kind of data from from categories and continuos values even including: 

* pixel data (images)
* zip codes 
* random data
* dates ...

The data often requires modification in order to fit into RF, but often just a minor modification (data engineering).

Most of the data engineering will be:

* take the log of some column as input instead of original data column
* convert your date columns into multiple convenient columns such as year, month, day, day of the week...
* convert categories (classes) to numbers ...

After preparing the data you will provide all the independent variables, and the dependent variable as a parameter to fit the RF.

The independent variables are multiple columns also called <strong>input samples</strong>, and the dependent variable is a single column or <strong>target</strong>. 

Simplified at the end RF will provide the score.

It may be the $R^2$ score also known as [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) for regression problems.

$R^2$ is the proportion of the variance in the dependent variable that is predictable from the independent variable(s).

$R^2$ can take values from -inf, to 1, where 1 is excellent, while 0 and anything less than 0 is considered very bad.

For RF and classification problems the score is the mean accuracy on the given test data.

>What is good when dealing with RF:

* RF doesn't assume your data distribution.
* RF doesn't assume the data relationship. 
* RF doesn't assume data interactions.
* RF is very hard to overfit


>Do we need a validation dataset dealing with RF?

It is best when we have a separate validation set, but we can get away even if we don't. We may use part of the test dataset as the validation dataset. Usually the validation set will take most recent data.


>Where to start with RF?

Probable one very nice starting point is to use the [scikit](https://scikit-learn.org).

For the regression tasks we can start with [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

For the classification tasks we can start with the [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

One nice thing I noticed with scikit RF, is you can do tasks in parallel (multiple processor support exists).


