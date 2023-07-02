# "MLRL-Common": Building-Blocks for Multi-label Rule Learning Algorithms 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/mlrl-common.svg)](https://badge.fury.io/py/mlrl-common)
[![Documentation Status](https://readthedocs.org/projects/mlrl-boomer/badge/?version=latest)](https://mlrl-boomer.readthedocs.io/en/latest/?badge=latest)

This software package provides common modules to be used by different types of **multi-label rule learning (MLRL)** algorithms that integrate with the popular [scikit-learn](https://scikit-learn.org) machine learning framework.

The goal of [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) is the automatic assignment of sets of labels to individual data points, for example, the annotation of text documents with topics.

The library serves as the basis for the implementation of the following rule learning algorithms:

* **BOOMER (Gradient Boosted Multi-label Classification Rules)**: A state-of-the art algorithm that uses [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) to learn an ensemble of rules that is built with respect to a given multivariate loss function.

## Functionalities

This package follows a unified and modular framework for the implementation of different types of MLRL algorithms. In the following, we provide an overview of the individual modules an instantiation of the framework must implement.

### Rule Induction

A module for rule induction that is responsible for the construction of individual rules. Currently, the following modules of this kind are implemented:

* A module for **greedy rule induction** that conducts a top-down search, where rules are constructed by adding one condition after the other and adjusting its prediction accordingly. 
* Rule induction based on a **beam search**, where a top-down search is conducted as described above. However, instead of focusing on the best solution at each step, the algorithm keeps track of a predefined number of promising solutions and picks the best one at the end.

All of the above modules support **numerical, ordinal, and nominal features** and can handle **missing feature values**. They can also be combined with methods for **unsupervised feature binning**, where training examples with similar features values are assigned to bins in order to reduce the training complexity. Moreover, **multi-threading** can be used to speed up training.

### Model Assemblage

A module for the assemblage of a rule model that consists of several rules. Currently, the following strategies can be used for constructing a model:

* **Sequential assemblage of rule models**, where one rule is learned after the other.

### Sampling Methods

A wide variety of sampling methods, including **sampling with and without replacement**, as well as **stratified sampling techniques**, is provided by this package. They can be used to learn new rules on a subset of the available training examples, features, or labels.

### (Label Space) Statistics

So-called label space statistics serve as the basis for assessing the quality of potential rules and determining their predictions. The notion of the statistics heavily depend on the rule learning algorithm at hand. For this reason, no particular implementation is currently included in this package.

### Post-Processing

Post-processing methods can be used to alter the predictions of a rule after it has been learned. Whether this is desirable or not heavily depends on the rule learning algorithm at hand. For this reason, no post-processing methods are currently provided by this package.

### Pruning Methods

Rule pruning techniques can optionally be applied to a rule after its construction to improve its generalization to unseen data and prevent overfitting. The following pruning techniques are currently supported by this package:

* **Incremental reduced error pruning (IREP)** removes overly specific conditions from a rule if this results in an increase of predictive performance (measured on a holdout set of the training data).

### Stopping Criteria

One or several stopping criteria can be used to decide whether additional rules should be added to a model or not. Currently, the following criteria are provided out-of-the-box:

* A **size-based stopping criterion** that ensures that a certain number of rules is not exceeded.
* A **time-based stopping criterion** that stops training as soon as a predefined runtime was exceeded.
* **Pre-pruning (a.k.a. early stopping)** aims at terminating the training process as soon as the performance of a model stagnates or declines (measured on a holdout set of the training data).

### Post-Optimization

Post-optimization methods can be employed to further improve the predictive performance of a model after it has been assembled. Currently, the following post-optimization techniques can be used:

* **Sequential post-optimization** reconstructs each rule in a model in the context of the other rules.

* **Post-pruning** may remove trailing rules from a model in this increases the models performance (as measured on a holdout set of the training data).

### Prediction algorithm

A prediction algorithm is needed to derive predictions from the rules in a previously assembled model. As prediction methods heavily depend on the rule learning algorithm at hand, no implementation is provided by this package out-of-the-box. However, it defines interfaces for the prediction of **regression scores, binary predictions, or probability estimates.**

## License

This project is open source software licensed under the terms of the [MIT license](../../../LICENSE.md). We welcome contributions to the project to enhance its functionality and make it more accessible to a broader audience. A frequently updated list of contributors is available [here](../../../CONTRIBUTORS.md). 
