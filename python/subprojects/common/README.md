# "MLRL-Common": Building-Blocks for Multi-label Rule Learning Algorithms 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/mlrl-common.svg)](https://badge.fury.io/py/mlrl-common)
[![Documentation Status](https://readthedocs.org/projects/mlrl-boomer/badge/?version=latest)](https://mlrl-boomer.readthedocs.io/en/latest/?badge=latest)

This software package provides common modules to be used by different types of **multi-label rule learning (MLRL)** algorithms that integrate with the popular [scikit-learn](https://scikit-learn.org) machine learning framework.

The goal of [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) is the automatic assignment of sets of labels to individual data points, for example, the annotation of text documents with topics.

The library serves as the basis for the implementation of the following rule learning algorithms:

* **BOOMER (Gradient Boosted Multi-label Classification Rules)**: A state-of-the art algorithm that uses [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) to learn an ensemble of rules that is built with respect to a given multivariate loss function.

## Features

This package follows a unified and modular framework for the implementation of different types of MLRL algorithms. An instantiation of the framework consists of the following modules:

* A module for **rule induction** that is responsible for the construction of individual rules. Each rule consists of a *body* and a *head*. The former specifies the region of the input space to which the rule applies. The latter provides predictions for one or several labels.  
* A strategy for the **assemblage of a rule model** that consists of several rules.
* A notion of **(label space) statistics** that serve as the basis for assessing the quality of potential rules and determining their predictions.
* Implementations of **pruning** techniques that can optionally be applied to a rule after its construction to improve the generalization to unseen data.
* **Post-processing** techniques that may alter the predictions of a rule after it has been learned.
* One or several **stopping criteria** that are used to decide whether more rules should be added to a model.
* Optional **sampling techniques** that may be used to obtain a subset of the available training examples, features or labels.
* An algorithm for the **aggregation of predictions** that are provided by the rules in a model for previously unseen *test examples*.

This library defines APIs for all the aforementioned modules and provides default implementations for the following ones:

* **Top-down hill climbing** for the greedy induction of rules. It supports numerical, ordinal and nominal features, as well as missing feature values. Optionally, a **histogram-based algorithm**, where training examples with similar feature values are assigned to bins, can be used to reduce the complexity of training. Both types of algorithms support the use of **multi-threading**.
* A strategy for the **sequential assemblage of rule models**, where one rule is learned after the other.
* **Incremental reduced error pruning (IREP)**, where conditions are removed from a rule's body if this results in increased performance as measured on a holdout set of the training data.
* **Simple stopping criteria** that stop the induction of rules after a certain amount of time or when a predefined number of rules has been reached, as well as an **early stopping mechanism** that allows to terminate training as soon as the performance of a model on a holdout set stagnates or declines.
* Methods for **sampling with or without replacement**, as well as **stratified sampling** techniques.

Furthermore, the library provides classes for the representation of individual rules, as well as dense and sparse data structures that may be used to store the feature values and ground truth labels of training and test examples.

## License

This project is open source software licensed under the terms of the [MIT license](https://github.com/mrapp-ke/Boomer/blob/master/LICENSE.txt). We welcome contributions to the project to enhance its functionality and make it more accessible to a broader audience. A frequently updated list of contributors is available [here](https://github.com/mrapp-ke/Boomer/blob/master/CONTRIBUTORS.md). 
