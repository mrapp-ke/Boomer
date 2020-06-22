# README

This project provides a scikit-learn implementation of "BOOMER" - an algorithm for learning gradient boosted multi-label classification rules.

The algorithm was first published in the following paper:

*Rapp M., Loza Mencía E., Fürnkranz J., Nguyen VL., Hüllermeier E. (2020) Learning Gradient Boosted Multi-label Classification Rules. In: Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2020. Lecture Notes in Computer Science. Springer, Cham*  

If you use the algorithm in a scientific publication, we would appreciate citations to the mentioned paper.

## Features

The algorithm that is provided by this project currently supports the following features to learn an ensemble of boosted classification rules:

* Different label-wise or example-wise loss functions can be minimized during training (optionally using L2 regularization).
* The rules may predict for a single label, or for all labels (which enables to model local label dependencies).
* When learning a new rule, random samples of the training examples, features or labels may be used, including different techniques such as sampling with or without replacement.
* The impact of individual rules on the ensemble can be controlled using shrinkage.
* The conditions of a recently induced rule can be pruned based on a hold-out set.  
* The algorithm currently only supports numerical or ordinal features. Nominal features can be handled by using one-hot encoding. 

## Project structure

```
|-- data                                Directory that contains several benchmark data sets
    |-- ...
|-- data-synthetic                      Directory that contains several synthetic data sets
    |-- ...
|-- python                              Directory that contains the project's source code
    |-- boomer                          Directory that contains the code for loading data sets and running experiments
        |-- algorithm                   Directory that contains the actual implementation of the learning algorithm 
            | ...
        | ...
    |-- main_boomer.py                  Can be used to start an experiment, i.e., to train and evaluate a model, using BOOMER
    |-- main_boomer_bbc_cv.py           Can be used to evaluate existing BOOMER models using "Bootstrap Bias Corrected Cross Validation" (BBC-CV)
    |-- main_boomer_plots.py            Can be used to plot visualizations of the performance of an existing BOOMER model
    |-- main_generate_synthetic_data.py Can be used to generate synthetic data sets
    |-- setup.py                        Distutil definition of the library for installation via pip
|-- slurm                               Directory that contains bash scripts for running jobs using the Slurm workload manager
    |-- ...
|-- Makefile                            Makefile for compiling the Cython source files and installing a Python virtual environment
|-- README.md                           This file
|-- settings.zip                        PyCharm settings for syntax highlighting of Cython code
```

## Project setup

The library provided by this project requires Python 3.7 and uses C extensions for Python using [Cython](https://cython.org) to speed up computation. It is recommended to create a virtual environment using the correct version of Python (which requires that this particular Python version is installed on the host) and providing all dependencies that are required to compile the Cython code (`numpy`, `scipy` and `Cython`). IDEs such as PyCharm may provide an option to create such a virtual environment automatically. For manual installation, the project comes with a Makefile that allows to create a virtual environment via the command
```
make venv
```  
This should create a new subdirectory `venv` within the project's root directory.

Unlike pure Python programs, the Cython source files (`.pyx` and `.pxd` files) that are used by the library must be compiled (see [documentation](http://docs.cython.org/en/latest/src/quickstart/build.html) for further details). The compilation can be started using the provided Makefile by running
```
make compile
```
This should result in `.c` files, as well as `.so` files (on Linux) or `.pyd` files (on Windows) be placed in the directory `python/boomer/algorithm/`.

To be able to use the library by any program run inside the virtual environment, it must be installed into the virtual environment together with all of its runtime dependencies (e.g. `scikit-learn`, a full list can be found in `setup.py`). For this purpose, the project's Makefile provides the command 

```
make install
```

*Whenever any Cython source files have been modified, they must be recompiled by running the command "make compile" again and updating the installed package via "make install" afterwards! If compiled Cython files do already exist, only the modified files will be recompiled.*

**Cleanup:** To get rid of any compiled C/C++ files, as well as of the virtual environment, the following command can be used:
```
make clean
``` 
For more fine-grained control, the command `make clean_venv` (for deleting the virtual environment) or `make clean_compile` (for deleting the compiled files) can be used.


**Syntax highlighting in PyCharm:** Unfortunately, the Community Edition of PyCharm does not come with Cython support. To enable proper syntax highlighting of Cython code, the file `settings.zip` in the project's root directory can be imported via `File -> Import Settings`.

## Running experiments

The file `main_boomer.py` allows to run experiments on a specific data set using different configurations of the learning algorithm. The implementation takes care of writing the experimental results into `.csv` files and the learned model can (optionally) be stored on disk to reuse it later. 

In order to run an experiment, the following command line arguments must be provided (most of them are optional):

| Parameter                    | Optional? | Default                    | Description                                                                                                                                                                                                                                                    |
|------------------------------|-----------|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--data-dir`                 | No        | `None`                     | The path of the directory where the data sets are located.                                                                                                                                                                                                     |
| `--dataset`                  | No        | `None`                     | The name of the data set file (without suffix).                                                                                                                                                                                                                |
| `--output-dir`               | Yes       | `None`                     | The path of the directory into which the experimental results (`.csv` files) should be written.                                                                                                                                                                |
| `--store-predictions`        | Yes       | `False`                    | `True`, if the predictions for the individual test examples should be stored as `.csv` files (they may become very large), `False` otherwise. Does only have an effect if the parameter `--output-dir` is specified.                                           |
| `--model-dir`                | Yes       | `None`                     | The path of the directory where models (`.model` files) are located.                                                                                                                                                                                           |
| `--parameter-dir`            | Yes       | `None`                     | The path of the directory, parameter settings (`.csv` files) should be loaded from.                                                                                                                                                                            |
| `--folds`                    | Yes       | `1`                        | The total number of folds to be used for cross-validation or `1`, if no cross validation should be used.                                                                                                                                                       |
| `--current-fold`             | Yes       | `-1`                       | The cross-validation fold to be performed or `-1`, if all folds should be performed. Must be `-1` or greater than `0`and less or equal to `--folds`. If `--folds` is 1, this parameter is ignored.                                                             |
| `--num-rules`                | Yes       | `500`                      | The number of rules to be induced or `-1`, if the number of rules should not be restricted.                                                                                                                                                                    |
| `--time-limit`               | Yes       | `-1`                       | The duration in seconds after which the induction of rules should be canceled or `-1`, if no time limit should be used.                                                                                                                                        |
| `--label-sub-sampling`       | Yes       | `-1`                       | The number of samples to be used for label sub-sampling. Must be at least1 or `-1`, if no sub-sampling should be used.                                                                                                                                         |
| `--instance-sub-sampling`    | Yes       | `None`                     | The name of the strategy to be used for instance sub-sampling. Must be `bagging`, `random-instance-selection` or `None`.                                                                                                                                       |
| `--feature-sub-sampling`     | Yes       | `None`                     | The name of the strategy to be used for feature sub-sampling. Must be `random-feature-selection` or `None`.                                                                                                                                                    |
| `--pruning`                  | Yes       | `None`                     | The name of the strategy to be used for pruning rules. Must be `irep` or `None`.                                                                                                                                                                               |
| `--head-refinement`          | Yes       | `None`                     | The name of the strategy to be used for finding the heads of rules. Must be `single-label`, `full` or `None`. If `None` is used, the most suitable strategy is chosen automatically based on the loss function.                                                |
| `--shrinkage`                | Yes       | `1.0`                      | The shrinkage parameter, a.k.a. the learning rate, to be used. Must be greater than `0` and less or equal to `1`.                                                                                                                                              |
| `--loss`                     | Yes       | `macro-squared-error-loss` | The name of the loss function to be minimized during training. Must be `macro-squared-error-loss` or `macro-logistic-loss` or `example-based-logistic-loss`.                                                                                                   |
| `--l2-regularization-weight` | Yes       | `0`                        | The weight of the L2 regularization that is applied for calculating the scores that are predicted by rules. Must be at least `0`. If `0` is used, the L2 regularization is turned off entirely, increasing the value causes the model to be more conservative. |
| `--random-state`             | Yes       | `1`                        | The seed to the be used by random number generators.                                                                                                                                                                                                           |
| `--log-level`                | Yes       | `info`                     | The log level to be used. Must be `debug`, `info`, `warn`, `warning`, `error`, `critical`, `fatal` or `notset`.                                                                                                                                                |

In the following, the command for running an experiment using an exemplary configuration can be seen. It uses a virtual environment as discussed in section "Project setup". 

```
venv/bin/python3.7 python/main_boomer.py --data-dir /path/to/data --output-dir /path/to/results/emotions --model-dir /path/to/models/emotions --dataset emotions --folds 10 --num-rules 1000 --instance-sub-sampling bagging --feature-sub-sampling random-feature-selection --loss macro-squared-error-loss --shrinkage 0.25 --pruning None --head-refinement single-label
```
