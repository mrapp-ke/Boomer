# "MLRL-Testbed": Utilities for the Evaluation of Multi-label Rule Learning Algorithms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/mlrl-testbed.svg)](https://badge.fury.io/py/mlrl-testbed)
[![Documentation Status](https://readthedocs.org/projects/mlrl-boomer/badge/?version=latest)](https://mlrl-boomer.readthedocs.io/en/latest/?badge=latest)

This software package provides **utilities for the training and evaluation of multi-label rule learning algorithms** that have been implemented using the "MLRL-Common" library, including the following ones:

* **BOOMER (Gradient Boosted Multi-label Classification Rules)**: A state-of-the art algorithm that uses [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) to learn an ensemble of rules that is built with respect to a given multivariate loss function.

## Features

Most notably, the package includes command line APIs that allow to configure the aforementioned algorithms, apply them to different datasets  and evaluate their predictive performance in terms of commonly used measures (provided by the [scikit-learn](https://scikit-learn.org/) framework). In addition, it provides the following functionality:

* The package supports **multi-label datasets in the [Mulan format](http://mulan.sourceforge.net/format.html)**.
* The predictive performance of algorithms can be evaluated using **predefined splits of a dataset** into training and test data or using **cross validation**.
* **Models can be saved into files** and reloaded for later use.
* **Parameter tuning can be used** to determine the optimal configuration of an algorithm and write it into a configuration file, which can afterwards be used for training a model.  
* **[Bootstrap Bias-corrected Cross Validation (BBC-CV)](https://arxiv.org/pdf/1708.07180v2.pdf)** can be used to incorporate parameter tuning and evaluation of an algorithm  in a computationally efficient way.
* **Textual representations of rule models**, as well as the **characteristics of models or datasets** can be written into output files.

## License

This project is open source software licensed under the terms of the [MIT license](https://github.com/mrapp-ke/Boomer/blob/master/LICENSE.txt). We welcome contributions to the project to enhance its functionality and make it more accessible to a broader audience. A frequently updated list of contributors is available [here](https://github.com/mrapp-ke/Boomer/blob/master/CONTRIBUTORS.md). 
