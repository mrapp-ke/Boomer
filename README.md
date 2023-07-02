<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo_light.svg">
    <img alt="BOOMER - Gradient Boosted Multi-Label Classification Rules" src="assets/logo_light.svg">
  </picture>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/mlrl-boomer.svg)](https://badge.fury.io/py/mlrl-boomer)
[![Documentation Status](https://readthedocs.org/projects/mlrl-boomer/badge/?version=latest)](https://mlrl-boomer.readthedocs.io/en/latest/?badge=latest)
[![Build](https://github.com/mrapp-ke/Boomer/actions/workflows/test_build.yml/badge.svg)](https://github.com/mrapp-ke/Boomer/actions/workflows/test_build.yml)
[![Code style](https://github.com/mrapp-ke/Boomer/actions/workflows/test_format.yml/badge.svg)](https://github.com/mrapp-ke/Boomer/actions/workflows/test_format.yml)
[![Twitter URL](https://img.shields.io/twitter/url?label=Follow%20on%20Twitter&style=social&url=https%3A%2F%2Ftwitter.com%2FBOOMER_ML)](https://twitter.com/BOOMER_ML)

**Important links:** [Documentation](https://mlrl-boomer.readthedocs.io) | [Issue Tracker](https://github.com/mrapp-ke/Boomer/issues) | [Changelog](CHANGELOG.md) | [Contributors](CONTRIBUTORS.md) | [Code of Conduct](CODE_OF_CONDUCT.md)

This software package provides the official implementation of **BOOMER - an algorithm for learning gradient boosted multi-label classification rules** that integrates with the popular [scikit-learn](https://scikit-learn.org) machine learning framework.

The goal of [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) is the automatic assignment of sets of labels to individual data points, for example, the annotation of text documents with topics. The BOOMER algorithm uses [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) to learn an ensemble of rules that is built with respect to a given multivariate loss function. To provide a versatile tool for different use cases, great emphasis is put on the *efficiency* of the implementation. To ensure its *flexibility*, it is designed in a modular fashion and can therefore easily be adjusted to different requirements.  

## References

The algorithm was first published in the following [paper](https://doi.org/10.1007/978-3-030-67664-3_8). A preprint version is publicly available [here](https://arxiv.org/pdf/2006.13346.pdf).

*Michael Rapp, Eneldo Loza Mencía, Johannes Fürnkranz Vu-Linh Nguyen and Eyke Hüllermeier. Learning Gradient Boosted Multi-label Classification Rules. In: Proceedings of the European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD), 2020, Springer.*

If you use the algorithm in a scientific publication, we would appreciate citations to the mentioned paper. An overview of publications that are concerned with the BOOMER algorithm, together with information on how to cite them, can be found in the section ["References"](https://mlrl-boomer.readthedocs.io/en/latest/references/index.html) of the documentation. 

## Functionalities

The algorithm that is provided by this project currently supports the following core functionalities to learn an ensemble of boosted classification rules:

* **Label-wise decomposable or non-decomposable loss functions** can be minimized in expectation.
* **L1 and L2 regularization** can be used.
* **Single-label, partial, or complete heads** can be used by rules, i.e., they can predict for an individual label, a subset of the available labels, or all labels. Predicting for multiple labels simultaneously enables rules to model local dependencies between labels.
* **Various strategies for predicting regression scores, labels or probabilities** are available.
* **Isotonic regression models can be used to calibrate marginal and joint probabilities** predicted by a model.
* **Rules can be constructed via a greedy search or a beam search.** The latter may help to improve the quality of individual rules.
* **Sampling techniques and stratification methods** can be used to learn new rules on a subset of the available training examples, features, or labels.
* **Shrinkage (a.k.a. the learning rate) can be adjusted** to control the impact of individual rules on the overall ensemble.
* **Fine-grained control over the specificity/generality of rules** is provided via hyper-parameters.
* **Incremental reduced error pruning** can be used to remove overly specific conditions from rules and prevent overfitting.
* **Post- and pre-pruning (a.k.a. early stopping)** allows to determine the optimal number of rules to be included in an ensemble.
* **Sequential post-optimization** may help to improve the predictive performance of a model by reconstructing each rule in the context of the other rules.
* **Native support for numerical, ordinal, and nominal features** eliminates the need for pre-processing techniques such as one-hot encoding.
* **Handling of missing feature values**, i.e., occurrences of NaN in the feature matrix, is implemented by the algorithm.

## Runtime and Memory Optimizations

In addition, the following features that may speed up training or reduce the memory footprint are currently implemented:

* **Unsupervised feature binning** can be used to speed up the evaluation of a rule's potential conditions when dealing with numerical features.
* **[Gradient-based label binning (GBLB)](https://arxiv.org/pdf/2106.11690.pdf)** can be used to assign the available labels to a limited number of bins. This may speed up training significantly when minimizing a non-decomposable loss function using rules with partial or complete heads.
* **Sparse feature matrices** can be used for training and prediction. This may speed up training significantly on some data sets.
* **Sparse label matrices** can be used for training. This may reduce the memory footprint in case of large data sets.
* **Sparse prediction matrices** can be used to store predicted labels. This may reduce the memory footprint in case of large data sets.
* **Sparse matrices for storing gradients and Hessians** can be used if supported by the loss function. This may speed up training significantly on data sets with many labels.
* **Multi-threading** can be used to parallelize the evaluation of a rule's potential refinements across several features, to update the gradients and Hessians of individual examples in parallel, or to obtain predictions for several examples in parallel.

## Documentation

An extensive user guide, as well as an API documentation for developers, is available at [https://mlrl-boomer.readthedocs.io](https://mlrl-boomer.readthedocs.io). If you are new to the project, you probably want to read about the following topics:

* Instructions for [installing the software package](https://mlrl-boomer.readthedocs.io/en/latest/quickstart/index.html#installation) or [building the project from source](https://mlrl-boomer.readthedocs.io/en/latest/api/index.html#building-from-source).
* Examples of how to [use the algorithm](https://mlrl-boomer.readthedocs.io/en/latest/quickstart/index.html#using-the-algorithm) in your own Python code or how to use the [command line API](https://mlrl-boomer.readthedocs.io/en/latest/testbed/index.html).
* An overview of available [parameters](https://mlrl-boomer.readthedocs.io/en/latest/quickstart/index.html#parameters).

A collection of benchmark datasets that are compatible with the algorithm are provided in a separate [repository](https://github.com/mrapp-ke/Boomer-Datasets).

For an overview of changes and new features that have been included in past releases, please refer to the [changelog](CHANGELOG.md).

## License

This project is open source software licensed under the terms of the [MIT license](LICENSE.md). We welcome contributions to the project to enhance its functionality and make it more accessible to a broader audience. A frequently updated list of contributors is available [here](CONTRIBUTORS.md).

All contributions to the project and discussions on the [issue tracker](https://github.com/mrapp-ke/Boomer/issues) are expected to follow the [code of conduct](CODE_OF_CONDUCT.md).
