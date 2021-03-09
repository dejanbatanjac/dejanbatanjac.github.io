---
published: false
layout: post
title: Ensembling
permalink: /ensembling
---
- [Bagging](#bagging)
- [Boosting](#boosting)
- [Stacking](#stacking)




## Bagging


Parameters that control bagging? 
• Changing the seed 
• Row (Sub) sampling or Bootstrapping 
• Shuffling 
• Column (Sub) sampling 
• Model-specific parameters 
• Number of models (or bags) 
• (Optionally) parallelism 

## Boosting

Boosting is a form of weighted averaging of models where each model is built sequentially via taking into account the past model performance.


Main boosting types 
• Weight based 
• Residual based 

id| f0 | f1 | f2 | f3 | y
---|----|----|---|---| ---
0| 0.84 |0.27 |0.72 | 0.43 | 1
1| 0.83 |0.79 |0.80 | 0.97 | 0
2| 0.74 |0.1  |0.89 | 0.34 | 0
3| 0.08 |0.26 |0.23 | 0.05 | 1
4| 0.71 |0.29 |0.03 | 0.42 | 0
5| 0.08 |0.76 |0.76 | 0.41 | 1


Weight based implementations:
• Sklearn's AdaBoost

AdaBoost (Adaptive Boosting) is a machine learning meta-algorithm.
Sklearn implementations:

* [for classification](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html){:rel="nofollow"}
* [for regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html){:rel="nofollow"}

Residual based implementations:
• Xgboost 
• Lightgbm 
• H20's GBM 
• Catboost
• Sklearn's GBM
    sklearn.ensemble.GradientBoostingClassifier
    sklearn.ensemble.GradientBoostingRegressor

[XGBoost](https://xgboost.ai/){:rel="nofollow"} is an open-source software library which provides a gradient boosting framework for C++, Java, Python, R, Julia, Perl, and Scala. It works on Linux, Windows, and macOS. From the project description, it aims to provide a "Scalable, Portable and Distributed Gradient Boosting Library"

[LightGBM](https://xgboost.ai/){:rel="nofollow"}, short for Light Gradient Boosting Machine, is a free and open source distributed gradient boosting framework for machine learning originally developed by Microsoft. It is based on decision tree algorithms and used for ranking, classification and other machine learning tasks.

[H2O's GBM](https://xgboost.ai/){:rel="nofollow"} (for Regression and Classification) is a forward learning ensemble method. The guiding heuristic is that good predictive results can be obtained through increasingly refined approximations. H2O's GBM sequentially builds regression trees on all the features of the dataset in a fully distributed way - each tree is built in parallel.

[CatBoost](https://xgboost.ai/){:rel="nofollow"} is an open-source software library developed by Yandex. It provides a gradient boosting framework which attempts to solve for Categorical features using a permutation driven alternative compared to the classical algorithm.

## Stacking
