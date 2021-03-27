# README

This project provides a scikit-learn implementation of "BOOMER" - an algorithm for learning gradient boosted multi-label classification rules.

The algorithm was first published in the following [paper](https://link.springer.com/chapter/10.1007/978-3-030-67664-3_8). A preprint version is publicly available [here](https://arxiv.org/pdf/2006.13346.pdf).

*Rapp M., Loza Mencía E., Fürnkranz J., Nguyen VL., Hüllermeier E. (2020) Learning Gradient Boosted Multi-label Classification Rules. In: Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2020. Lecture Notes in Computer Science, pp. 124-140, vol 12459. Springer, Cham*  

If you use the algorithm in a scientific publication, we would appreciate citations to the mentioned paper.

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
* Dense or sparse feature matrices can be used for training and prediction. The use of sparse matrices may speed-up training significantly on some data sets.
* Dense or sparse label matrices can be used for training. The use of sparse matrices may reduce the memory footprint in case of large data sets.
* Multi-threading can be used to parallelize the evaluation of a rule's potential refinements across multiple CPU cores. 

## Project structure

```
|-- data                                Contains several benchmark data sets
    |-- ...
|-- data-synthetic                      Contains several synthetic data sets
    |-- ...
|-- cpp                                 Contains the implementation of core algorithms in C++
    |-- subprojects
        |-- common                      Contains implementations that all algorithms have in common
        |-- boosting                    Contains implementations of boosting algorithms
        |-- seco                        Contains implementations of separate-and-conquer algorithms
    |-- ...
|-- python                              Contains Python code for running experiments using different algorithms
    |-- mlrl
        |-- common                      Contains Python code that is needed to run any kind of algorithms
            |-- cython                  Contains commonly used Cython wrappers
            |-- ...
        |-- boosting                    Contains Python code for running boosting algorithms
            |-- cython                  Contains boosting-specific Cython wrappers
            |-- ...
        |-- seco                        Contains Python code for running separate-and-conquer algorithms
            |-- cython                  Contains separate-and-conquer-specific Cython wrappers
            |-- ...
        |-- testbed                     Contains useful functionality for running experiments, e.g., for cross validation, writing of output files, etc.
            |-- ...
    |-- main_boomer.py                  Can be used to start an experiment using the BOOMER algorithm
    |-- main_seco.py                    Can be used to start an experiment using the separate-and-conquer algorithm
    |-- main_generate_synthetic_data.py Can be used to generate synthetic data sets
    |-- ...
|-- Makefile                            Makefile for compilation
|-- ...
```

## Project setup

The algorithm provided by this project is implemented in C++. In addition, a Python wrapper that implements the scikit-learn API is available. To be able to integrate the underlying C++ implementation with Python, [Cython](https://cython.org) is used.

The C++ implementation, as well as the Cython wrappers, must be compiled in order to be able to run the provided algorithm. To facilitate compilation, this project comes with a Makefile that automatically executes the necessary steps.

At first, a virtual Python environment can be created via the following command:
```
make venv
```

As a prerequisite, Python 3.7 (or a more recent version) must be available on the host system. All compile-time dependencies (`numpy`, `scipy`, `Cython`, `meson` and `ninja`) that are required for building the project will automatically be installed into the virtual environment. As a result of executing the above command, a subdirectory `venv` should have been created within the project's root directory.

Afterwards, the compilation can be started by executing the following command:
```
make compile
```

Finally, the library must be installed into the virtual environment, together with all of its runtime dependencies (e.g. `scikit-learn`, a full list can be found in `setup.py`). For this purpose, the project's Makefile provides the following command:

```
make install
```

*Whenever any C++ or Cython source files have been modified, they must be recompiled by running the command `make compile` again! If compilation files do already exist, only the modified files will be recompiled.*

**Cleanup:** To get rid of any compilation files, as well as of the virtual environment, the following command can be used:
```
make clean
``` 

For more fine-grained control, the command `make clean_venv` (for deleting the virtual environment) or `make clean_compile` (for deleting the compiled files) can be used. If only the compiled Cython files should be removed, the command `make clean_cython` can be used. Accordingly, the command `make clean_cpp` removes the compiled C++ files.

**Syntax highlighting in PyCharm:** Unfortunately, the Community Edition of PyCharm does not come with Cython support. To enable proper syntax highlighting of Cython code, the file `settings.zip` in the project's root directory can be imported via `File -> Import Settings`.

## Running experiments

The file `main_boomer.py` allows to run experiments on a specific data set using different configurations of the learning algorithm. The implementation takes care of writing the experimental results into `.csv` files and the learned model can (optionally) be stored on disk to reuse it later. 

In order to run an experiment, the following command line arguments must be provided (most of them are optional):

| Parameter                    | Optional? | Default                    | Description                                                                                                                                                                                                                                                                                                                                                                                                           |
|------------------------------|-----------|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--data-dir`                 | No        | `None`                     | The path of the directory where the data sets are located.                                                                                                                                                                                                                                                                                                                                                            |
| `--dataset`                  | No        | `None`                     | The name of the data set file (without suffix).                                                                                                                                                                                                                                                                                                                                                                       |
| `--one-hot-encoding`         | Yes       | `False`                    | `True`, if one-hot-encoding should be used for nominal attributes, `False` otherwise.                                                                                                                                                                                                                                                                                                                                 |
| `--output-dir`               | Yes       | `None`                     | The path of the directory into which the experimental results (`.csv` files) should be written.                                                                                                                                                                                                                                                                                                                       |
| `--store-predictions`        | Yes       | `False`                    | `True`, if the predictions for individual examples and labels should be stored as `.arff` files, `False` otherwise. Does only have an effect if the parameter `--output-dir` is specified.                                                                                                                                                                                                                            |
| `--print-rules`              | Yes       | `False`                    | `True`, if the induced rules should be printed on the console, `False` otherwise.                                                                                                                                                                                                                                                                                                                                     |
| `--store-rules`              | Yes       | `False`                    | `True`, if the induced rules should be stored as a `.txt` file, `False` otherwise. Does only have an effect if the parameter `--output-dir` is specified.                                                                                                                                                                                                                                                             |
| `--print-options`            | Yes       | `None`                     | A dictionary that specifies additional options to be used for printing or storing rules, if the parameter `--print-rules` and/or `--store-rules` is set to `True`, e.g. `{'print_nominal_values':True}`.                                                                                                                                                                                                              |
| `--evaluate-training-data`   | Yes       | `False`                    | `True`, if the models should not only be evaluated on the test data, but also on the training data, `False` otherwise.                                                                                                                                                                                                                                                                                                |
| `--model-dir`                | Yes       | `None`                     | The path of the directory where models (`.model` files) are located.                                                                                                                                                                                                                                                                                                                                                  |
| `--parameter-dir`            | Yes       | `None`                     | The path of the directory, parameter settings (`.csv` files) should be loaded from.                                                                                                                                                                                                                                                                                                                                   |
| `--folds`                    | Yes       | `1`                        | The total number of folds to be used for cross-validation or `1`, if no cross validation should be used.                                                                                                                                                                                                                                                                                                              |
| `--current-fold`             | Yes       | `-1`                       | The cross-validation fold to be performed or `-1`, if all folds should be performed. Must be `-1` or greater than `0`and less or equal to `--folds`. If `--folds` is 1, this parameter is ignored.                                                                                                                                                                                                                    |
| `--max-rules`                | Yes       | `1000`                     | The number of rules to be induced or `-1`, if the number of rules should not be restricted.                                                                                                                                                                                                                                                                                                                           |
| `--time-limit`               | Yes       | `-1`                       | The duration in seconds after which the induction of rules should be canceled or `-1`, if no time limit should be used.                                                                                                                                                                                                                                                                                               |
| `--early-stopping`           | Yes       | `None`                     | The name of the strategy to be used for early stopping. Must be `measure` or `None`, if no early stopping should be used. Additional arguments may be provided as a dictionary, e.g., `measure{'min_rules':100,'update_interval':1,'stop_interval':1,'num_past':50,'num_recent':50,'aggregation':'min','tolerance':0.001}`. Does only have an effect if the parameter `--holdout` is set to a value greater than `0`. |                                                                                
| `--label-sub-sampling`       | Yes       | `None`                     | The name of the strategy to be used for label sub-sampling. Must be `random-label-selection` or `None`. Additional arguments may be provided as a dictionary, e.g. `random-label-selection{'num_samples':5}`.                                                                                                                                                                                                         |                                                       
| `--instance-sub-sampling`    | Yes       | `bagging`                  | The name of the strategy to be used for instance sub-sampling. Must be `bagging`, `random-instance-selection` or `None`. Additional arguments may be provided as a dictionary, e.g. `bagging{'sample_size':0.5}`.                                                                                                                                                                                                     |
| `--feature-sub-sampling`     | Yes       | `random-feature-selection` | The name of the strategy to be used for feature sub-sampling. Must be `random-feature-selection` or `None`. Additional arguments may be provided as a dictionary, e.g. `random_feature-selection{'sample_size':0.5}`.                                                                                                                                                                                                 |
| `--holdout`                  | Yes       | `0`                        | The fraction of the training examples that should be included in the holdout set. Must be in greater than `0` and smaller than `1`  or `0`, if no holdout set should be used.                                                                                                                                                                                                                                         |
| `--feature-binning`          | Yes       | `None`                     | The name of the strategy to be used for feature binning. Must be `equal-width`, `equal-frequency` or `None`, if no feature binning should be used. Additional arguments may be provided as a dictionary, e.g. `equal-width{'bin_ratio':0.5,'min_bins':2,'max_bins':256}`.                                                                                                                                             |
| `--pruning`                  | Yes       | `None`                     | The name of the strategy to be used for pruning rules. Must be `irep` or `None`. Does only have an effect if the parameter `--instance-sub-sampling` is not set to `None`.                                                                                                                                                                                                                                            |
| `--min-coverage`             | Yes       | `1`                        | The minimum number of training examples that must be covered by a rule. Must be at least `1`.                                                                                                                                                                                                                                                                                                                         |
| `--max-conditions`           | Yes       | `-1`                       | The maximum number of conditions to be included in a rule's body. Must be at least `1` or `-1`, if the number of conditions should not be restricted.                                                                                                                                                                                                                                                                 |
| `--max-head-refinements`     | Yes       | `-1`                       | The maximum number of times the head of a rule may be refined. Must be at least `1` or `-1`, if the number of refinements should not be restricted.                                                                                                                                                                                                                                                                   |
| `--head-refinement`          | Yes       | `None`                     | The name of the strategy to be used for finding the heads of rules. Must be `single-label`, `full` or `None`. If `None` is used, the most suitable strategy is chosen automatically based on the loss function.                                                                                                                                                                                                       |
| `--shrinkage`                | Yes       | `0.3`                      | The shrinkage parameter, a.k.a. the learning rate, to be used. Must be greater than `0` and less or equal to `1`.                                                                                                                                                                                                                                                                                                     |
| `--loss`                     | Yes       | `label-wise-logistic-loss` | The name of the loss function to be minimized during training. Must be `label-wise-squared-error-loss`, `label-wise-squared-hinge-loss`, `label-wise-logistic-loss` or `example-wise-logistic-loss`.                                                                                                                                                                                                                  |
| `--predictor`                | Yes       | `None`                     | The name of the strategy to be used for making predictions. Must `label-wise`, `example-wise` or `None`. If `None` is used, the most suitable strategy is chosen automatically based on the loss function.                                                                                                                                                                                                            |
| `--l2-regularization-weight` | Yes       | `1.0`                      | The weight of the L2 regularization that is applied for calculating the scores that are predicted by rules. Must be at least `0`. If `0` is used, the L2 regularization is turned off entirely, increasing the value causes the model to be more conservative.                                                                                                                                                        |
| `--random-state`             | Yes       | `1`                        | The seed to the be used by random number generators.                                                                                                                                                                                                                                                                                                                                                                  |
| `--feature-format`           | Yes       | `auto`                     | The format to be used for the feature matrix. Must be `sparse`, if a sparse matrix should be used, `dense`, if a dense matrix should be used, or `auto`, if the format should be chosen automatically.                                                                                                                                                                                                                |
| `--label-format`             | Yes       | `auto`                     | The format to be used for the label matrix. Must be `sparse`, if a sparse matrix should be used, `dense`, if a dense matrix should be used, or `auto`, if the format should be chosen automatically.                                                                                                                                                                                                                  |
| `--num-threads-refinement`   | Yes       | `1`                        | The number of threads to be used to search for potential refinements of rules. Must be at least `1` or `-1`, if the number of cores that are available on the machine should be used.                                                                                                                                                                                                                                 |
| `--num-threads-update`       | Yes       | `1`                        | The number of threads to be used to calculate gradients and Hessians. Must be at least `1` or `-1`, if the number of cores that are available on the machine should be used.                                                                                                                                                                                                                                          |
| `--num-threads-prediction`   | Yes       | `1`                        | The number of threads to be used to make predictions. Must be at least `1` or `-1`, if the number of cores that are available on the machine should be used.                                                                                                                                                                                                                                                          |
| `--log-level`                | Yes       | `info`                     | The log level to be used. Must be `debug`, `info`, `warn`, `warning`, `error`, `critical`, `fatal` or `notset`.                                                                                                                                                                                                                                                                                                       |

In the following, the command for running an experiment using an exemplary configuration can be seen. It uses a virtual environment as discussed in section "Project setup". 

```
venv/bin/python3 python/main_boomer.py --data-dir /path/to/data --output-dir /path/to/results/emotions --model-dir /path/to/models/emotions --dataset emotions --folds 10 --num-rules 1000 --instance-sub-sampling bagging --feature-sub-sampling random-feature-selection --loss label-wise-logistic-loss --shrinkage 0.3 --pruning None --head-refinement single-label
```
