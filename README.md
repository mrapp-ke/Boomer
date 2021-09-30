<p align="center">
  <img alt="BOOMER - Gradient Boosted Multi-Label Classification Rules" src="https://github.com/mrapp-ke/Boomer-Development/blob/development/logo.png" height="128"/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/mlrl-boomer/badge/?version=latest)](https://mlrl-boomer.readthedocs.io/en/latest/?badge=latest)

This project provides a scikit-learn implementation of **BOOMER - an algorithm for learning gradient boosted multi-label classification rules**.

The goal of [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) is the automatic assignment of sets of labels to individual data points, for example, the annotation of text documents with topics. The BOOMER algorithm uses [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) to learn an ensemble of rules that is built with respect to a given multivariate loss function. To provide a versatile tool for different use cases, great emphasis is put on the *efficiency* of the implementation. To ensure its *flexibility*, it is designed in a modular fashion and can therefore easily be adjusted to different requirements.  

## References

The algorithm was first published in the following [paper](https://doi.org/10.1007/978-3-030-67664-3_8). A preprint version is publicly available [here](https://arxiv.org/pdf/2006.13346.pdf).

*Michael Rapp, Eneldo Loza Mencía, Johannes Fürnkranz Vu-Linh Nguyen and Eyke Hüllermeier. Learning Gradient Boosted Multi-label Classification Rules. In: Proceedings of the European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD), 2020, Springer.*

If you use the algorithm in a scientific publication, we would appreciate citations to the mentioned paper. An overview of publications that are concerned with the BOOMER algorithm, together with information on how to cite them, can be found in the section [References](https://mlrl-boomer.readthedocs.io/en/latest/references/index.html) of the documentation. 

## Features

The algorithm that is provided by this project currently supports the following core functionalities to learn an ensemble of boosted classification rules:

* Different label-wise or example-wise loss functions can be minimized during training (optionally using L2 regularization).
* The rules may predict for a single label or for all labels (which enables to model local label dependencies).
* When learning a new rule, random samples of the training examples, features or labels may be used (including different techniques such as sampling with or without replacement).
* The impact of individual rules on the ensemble can be controlled using shrinkage.
* Hyper-parameters that provide fine-grained control over the specificity/generality of rules are available.
* The conditions of rules can be pruned based on a hold-out set.
* The algorithm can natively handle numerical, ordinal and nominal features (without the need for pre-processing techniques such as one-hot encoding).
* The algorithm is able to deal with missing feature values, i.e., occurrences of NaN in the feature matrix.
* Different strategies for prediction, which can be tailored to the used loss function, are available.

In addition, the following features that may speed up training or reduce the memory footprint are currently implemented:

* Approximate methods for evaluating potential conditions of rules, based on unsupervised binning methods, can be used.
* [Gradient-based label binning (GBLB)](https://arxiv.org/pdf/2106.11690.pdf) can be used to assign the available labels to a limited number of bins. The use of label binning may speed up training significantly when using rules that predict for multiple labels to minimize a non-decomposable loss function.
* Dense or sparse feature matrices can be used for training and prediction. The use of sparse matrices may speed up training significantly on some data sets.
* Dense or sparse label matrices can be used for training. The use of sparse matrices may reduce the memory footprint in case of large data sets.
* Dense or sparse matrices can be used to store predictions. The use of sparse matrices may reduce the memory footprint in case of large data sets.
* Multi-threading can be used to parallelize the evaluation of a rule's potential refinements across multiple CPU cores. 

## Documentation

An extensive user guide, as well as an API documentation for developers, is available at [https://mlrl-boomer.readthedocs.io](https://mlrl-boomer.readthedocs.io). If you are new to the project, you probably want to read about the following topics:

* Instructions on [building the project](https://mlrl-boomer.readthedocs.io/en/latest/quickstart/index.html#building-the-project).
* [Examples](https://mlrl-boomer.readthedocs.io/en/latest/quickstart/index.html#running-the-algorithm) of how the algorithm can be run.
* An overview of available [parameters](https://mlrl-boomer.readthedocs.io/en/latest/quickstart/index.html#parameters).

For an overview of changes and new features that have been included in past releases, please refer to the [changelog](CHANGELOG.md).

## License

This project is open source software licensed under the terms of the [MIT license](LICENSE.txt). We welcome contributions to the project to enhance its functionality and make it more accessible to a broader audience. A frequently updated list of contributors is available [here](CONTRIBUTORS.md). 
