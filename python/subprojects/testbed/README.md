# "MLRL-Testbed": Utilities for Evaluating Multi-label Rule Learning Algorithms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/mlrl-testbed.svg)](https://badge.fury.io/py/mlrl-testbed)
[![Documentation Status](https://readthedocs.org/projects/mlrl-boomer/badge/?version=latest)](https://mlrl-boomer.readthedocs.io/en/latest/?badge=latest)

This software package provides **utilities for training and evaluating single- and multi-label rule learning algorithms** that have been implemented using the "MLRL-Common" library, including the following ones:

* **BOOMER (Gradient Boosted Multi-label Classification Rules)**: A state-of-the art algorithm that uses [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) to learn an ensemble of rules that is built with respect to a given multivariate loss function.

## Functionalities

Most notably, the package includes command line APIs that allow configuring the algorithms mentioned above, applying them to different datasets, and evaluating their predictive performance in terms of commonly used measures (provided by the [scikit-learn](https://scikit-learn.org/) framework). In summary, it provides the following functionalities:

* **Sinle- and multi-label datasets in the [Mulan](http://mulan.sourceforge.net/format.html) and [Meka format](https://waikato.github.io/meka/datasets/)** are supported.
* **Datasets can automatically be split into training and test data, including the possibility to use cross validation.** Alternatively, predefined splits can be used by supplying the data as separate files.
* **One-hot-encoding** can be applied to nominal or binary features.
* **Binary predictions, regression scores, or probability estimates** can be obtained from a model. Evaluation measures that are suited for the respective type of predictions are picked automatically.
* **Evaluation scores can be saved** to output files and printed on the console.
* **Rule models can be evaluated incrementally**, i.e., they can be evaluated repeatedly using a subset of the rules with increasing size.
* **Textual representations of rule models can be saved** to output files and printed on the console. In addition, the **characteristics of models can also be saved** and printed.
* **Characteristics of datasets can be saved** to output files and printed on the console.
* **Unique label vectors contained in a dataset can be saved** to output files and printed on the console.
* **Predictions can be saved** to output files and printed on the console. In addition, **characteristics of binary predictions can also be saved** and printed.
* **Models for the calibration of probabilities can be saved** to output files and printed on the console.
* **Models can be saved on disk** in order to be reused by future experiments.
* **Algorithmic parameters can be read from configuration files** instead of providing them via command line arguments. When providing parameters via the command line, corresponding configuration files can automatically be saved on disk.

## License

This project is open source software licensed under the terms of the [MIT license](../../../LICENSE.md). We welcome contributions to the project to enhance its functionality and make it more accessible to a broader audience. A frequently updated list of contributors is available [here](../../../CONTRIBUTORS.md). 
