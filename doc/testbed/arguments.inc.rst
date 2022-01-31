.. _arguments:

Command Line Arguments
----------------------

In addition to the mandatory arguments that must be provided to the command line API to specify the dataset to be used for training, a wide variety of optional arguments are available as well. In the following, we provide an overview of these arguments and discuss their respective purposes.

* ``--folds`` (Default value = ``1``)

  * The total number of folds to be used for cross validation. Must be greater than 1 or 1, if no cross validation should be used. If set to 1, the program will try to use predefined splits of a dataset into training and test data. Given that ``name`` is provided as the value of the argument ``--dataset``, the former must be provided as a file named ``name-train.arff``, whereas the latter is retrieved from a file with the name ``name-test.arff``. If no such files are available or if cross validation should be used, the program will look for a file with the name ``name.arff`` instead.

* ``--current-fold`` (Default value = ``0``)

  * The cross validation fold to be performed. Must be in [1, ``--folds``] or 0, if all folds should be performed. This parameter is ignored if the argument ``--folds`` is set to 1.

* ``--evaluate-training-data`` (Default value = ``false``)

  * ``true`` The models are not only evaluated on the test data, but also on the training data.
  * ``false`` The models are only evaluated on the test data.

* ``--one-hot-encoding`` (Default value = ``false``)

  * ``true`` One-hot-encoding is used to encode nominal attributes.
  * ``false`` The algorithm's ability to natively handle nominal attributes is used.

* ``--model-dir`` (Default value = ``None``)

  * The path of the directory where models should be stored. If such models are found in the specified directory, they will be used instead of learning a new model from scratch. If no models are available, the trained models will be saved in the specified directory once training has completed.

* ``--parameter-dir`` (Default value = ``None``)

  * The path of the directory where configuration files, which specify the parameters to be used by the algorithm, are located. If such files are found in the specified directory, the specified parameter settings are used instead of the parameters that are provided via command line arguments.

* ``--output-dir`` (Default value = ``None``)

  * The path of the directory where experimental results should be saved.

* ``--print-evaluation`` (Default value = ``true``)

  * ``true`` The evaluation results in terms of common metrics are printed on the console.
  * ``false`` The evaluation results are not printed on the console.

* ``--store-evaluation`` (Default value = ``true``)

  * ``true`` The evaluation results in terms of common metrics are written into .csv files. Does only have an effect if the parameter ``--output-dir`` is specified.
  * ``false`` The evaluation results are not written into .csv files.

* ``--store-predictions`` (Default value = ``false``)

  * ``true`` The predictions for individual examples and labels are written into .arff files. Does only have an effect if the parameter ``--output-dir`` is specified.
  * ``false`` Predictions are not written into .arff files.

* ``--print-data-characteristics`` (Default value = ``false``)

  * ``true`` The characteristics of the training data set are printed on the console
  * ``false`` The characteristics of the training data set are not printed on the console

* ``--store-data-characteristics`` (Default value = ``false``)

  * ``true`` The characteristics of the training data set are written into a .csv file. Does only have an effect if the parameter ``--output-dir`` is specified.
  * ``false`` The characteristics of the training data set are not written into a .csv file.

* ``--print-model-characteristics`` (Default value = ``false``)

  * ``true`` The characteristics of rule models are printed on the console
  * ``false`` The characteristics of rule models are not printed on the console

* ``--store-model-characteristics`` (Default value = ``false``)

  * ``true`` The characteristics of rule models are written into a .csv file. Does only have an effect if the parameter ``--output-dir`` is specified.
  * ``false`` The characteristics of rule models are not written into a .csv file.

* ``--print-rules`` (Default value = ``false``)

  * ``true`` The induced rules are printed on the console.
  * ``false`` The induced rules are not printed on the console.

* ``--store-rules`` (Default value = ``false``)

  * ``true`` The induced rules are written into a .txt file. Does only have an effect if the parameter ``--output-dir`` is specified.
  * ``false`` The induced rules are not written into a .txt file.

* ``--print-options`` (Default value = ``None``)

  * Additional options to be taken into account when writing rules on the console or into an output file. Does only have an effect, if the parameter ``--print-rules`` or ``--store-rules`` is set to ``true``. The options must be given using the bracket notation (see :ref:`parameters`). The following options are available:

    * ``print_feature_names`` (Default value = ``true``) ``true``, if the names of features should be printed instead of their indices, ``false`` otherwise.
    * ``print_label_names`` (Default value = ``true``) ``true``, if the names of labels should be printed instead of their indices, ``false`` otherwise.
    * ``print_nominal_values`` (Default value = ``true``) ``true``, if the names of nominal values should be printed instead of their numerical representation, ``false`` otherwise.

* ``--log-level`` (Default value = ``info``)

  * The log level to be used. Must be ``debug``, ``info``, ``warn``, ``warning``, ``error``, ``critical``, ``fatal`` or ``notset``.

Overwriting Algorithmic Parameters
----------------------------------

In addition to the command line arguments that are discussed above, it is often desirable to not use the default configuration of the BOOMER algorithm in an experiment, but to customize some of its parameters. For this purpose, all of the algorithmic parameters that are discussed in the section :ref:`parameters` may be overwritten by providing corresponding arguments to the command line API.

To be in accordance with the syntax that is typically used by command line programs, the parameter names must be given according to the following syntax that slightly differs from the names that are used by the programmatic Python API:

* All argument names must start with two leading dashes (``--``).
* Underscores (``_``) must be replaced with dashes (``-``).

For example, the value of the parameter ``feature_binning`` may be overwritten as follows:

.. code-block:: text

   boomer --data-dir /path/to/datasets/ --dataset name --feature-binning equal-width

Some algorithmic parameters, including the parameter ``feature_binning``, allow to specify additional options as key-value pairs by using a bracket notation. This is also supported by the command line API, where the options may not contain any spaces and special characters like ``{`` or ``}`` must be escaped by using single-quotes (``'``):

.. code-block:: text

   boomer --data-dir /path/to/datasets/ --dataset name --feature-binning equal-width'{bin_ratio=0.33,min_bins=2,max_bins=64}'
