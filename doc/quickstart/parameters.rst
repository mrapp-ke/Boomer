Parameters
----------

The behavior of the BOOMER algorithm can be controlled in a fine-grained manner via a large number of parameters. Most of these parameters are optional. If not specified otherwise, default settings that work well in most of the cases are used. In the following, an overview of all available parameters, as well as their default values, is provided.

.. note::
    Each parameter is identified by an unique name and must be specified according to the following syntax:

    ``--parameter-name value``

    In addition to the specified value, some parameters allow to specify additional options as key-value pairs. These options may be provided by using the following bracket notation:

    ``--parameter-name value{key1=value1,key2=value2}``

    Parameter values that include additional options may not contain any spaces. Depending on the shell that is used to run the program, special characters like ``{`` or ``}`` must eventually be escaped. When using bash or sh this can be achieved by adding single quotes as follows:

    ``--parameter-name value'{key1=value1,key2=value2}'``.

**Data set**

The following parameters are always needed to specify the data set that should be used for training:

* ``--data-dir``

  * The path of the directory where the data set files are located (an ARFF file and a corresponding XML file according to the Mulan format).

* ``--dataset``

  * The name of the data set files (without suffix).

**Training/Testing Procedure**

* ``--folds`` (Default value = 1)

  * The total number of folds to be used for cross validation. Must be greater than 1 or 1, if no cross validation should be used.

* ``--current-fold`` (Default value = 0)

  * The cross validation fold to be performed. Must be in [1, --folds] or 0, if all folds should be performed. This parameter is ignored if --folds is set to 1.

* ``--evaluate-training-data`` (Default value = false)

  * ``true`` The models are not only evaluated on the test data, but also on the training data.
  * ``false`` The models are only evaluated on the test data.

**Data Format**

The following parameters allow to specify how the training data should be organized:

* ``--one-hot-encoding`` (Default value = false)

  * ``true`` One-hot-encoding is used to encode nominal attributes.
  * ``false`` The algorithm's ability to natively handle nominal attributes is used.

* ``--feature-format`` (Default value = auto)

  * ``auto`` The most suitable format for representation of the feature matrix is chosen automatically by estimsting which representation requires less memory.
  * ``dense`` Enforces that the feature matrix is stored using a dense format. 
  * ``sparse`` Enforces that the feature matrix is stored using a sparse format. Using a sparse format may reduce the memory footprint and/or speed up the training process on some data sets.

* ``--label-format`` (Default value = auto)

  * ``auto`` The most suitable format for representation of the label matrix is chosen automatically by estimating which representation requires less memory.
  * ``dense`` Enforces that the label matrix is stored using a dense format.
  * ``sparse`` Enforces that the label matrix is stored using a sparse format. Using a sparse format may reduce the memory footprint on some data sets.

* ``--prediction-format`` (Default value = auto)

  * ``auto`` The most suitable format for representation of predicted labels is chosen automatically based on the sparsity of the ground truth labels supplied for training.
  * ``dense`` Enforces that predicted labels are stored using a dense format.
  * ``sparse`` Enforces that predicted labels are stored using a sparse format. Using a sparse format may reduce the memory footprint on some data sets.

**Input Files**

The following parameters allow to specify the directories, where input files can be found:

* ``--model-dir`` (Default value = None)

  * The path of the directory where models should be stored. If such models are found in the specified directory, they will be used instead of training from scratch. If no models are available, the trained models will be saved in the specified directory once training has completed.

* ``--parameter-dir`` (Default value = None)

  * The path of the directory where configuration files, which specify the parameters to be used by the algorithm, are located. If such files are found in the specified directory, the specified parameter settings are used instead of the parameters that are provided via command line parameters.

**Output**

The following parameters allow to customize the console output and output files that are written by the algorithm:

* ``--output-dir`` (Default value = None)

  * The path of the directory where experimental results should be saved.

* ``--store-predictions`` (Default value = false)

  * ``true`` The predictions for individual examples and labels are written into output files. Does only have an effect if the parameter --output-dir is specified.
  * ``false`` Predictions are not written into output files.

* ``--print-rules`` (Default value = false)

  * ``true`` The induced rules are printed on the console.
  * ``false`` The induced rules are not printed on the console.

* ``--store-rules`` (Default value = false)

  * ``true`` The induced rules are written into a text file. Does only have an effect if the parameter --output-dir is specified.
  * ``false`` The induced rules are not written into a text file.

* ``--print-options`` (Default value = None)

  * Additional options to be taken into account when writing rules on the console or into an output file. Does only have an effect, if the parameter --print-rules or --store-rules is set to ``true``. The options must be given using the bracket notation. The following options are available:
  
    * ``print_feature_names`` (Default value = true) ``true``, if the names of features should be printed instead of their indices, ``false`` otherwise.
    * ``print_label_names`` (Default value = true) ``true``, if the names of labels should be printed instead of their indices, ``false`` otherwise.
    * ``print_nominal_values`` (Default value = true) ``true``, if the names of nominal values should be printed instead of their numerical representation, ``false`` otherwise.

* ``--log-level`` (Default value = info)

  * The log level to be used. Must be debug, info, warn, warning, error, critical, fatal or notset.


**Algorithmic Parameters**

The following parameters allow to adjust the behavior of the algorithm:

* ``--random-state`` (Default value = 1)

  * The seed to be used by random number generators. Must be at least 1.

* ``--max-rules`` (Default value = 1000)

  * The maximum number of rules to be induced. Must be at least 1 or 0, if the number of rules should not be restricted.

* ``--default-rule`` (Default value = true)

  * ``true`` The first rule is a default rule.
  * ``false`` No default rule is used.

* ``--time-limit`` (Default value = 0)

  * The duration in seconds after which the induction of rules should be canceled. Must be at least 1 or 0, if no time limit should be set.

* ``--label-sampling`` (Default value = None)

  * ``None`` All labels are considered for learning a new rule.
  * ``without-replacement`` The labels to be considered when learning a new rule are chosen randomly. The following options may be provided using the bracket notation:
  
    * ``num_samples`` (Default value = 1) The number of labels the be included in a sample. Must be at least 1.

* ``--feature-sampling`` (Default value = without-replacement)

  * ``None`` All features are considered for learning a new rule.
  * ``without-replacement`` A random subset of the features is used to search for the refinements of rules. The following options may be provided using the bracket notation:

    * ``sample_size`` (Default value = 0) The percentage of features to be included in a sample, e.g., a value of 0.6 corresponds to 60% of the features. Must be in (0, 1] or 0, if the sample size should be calculated as log2(numFeatures - 1) + 1).

* ``--instance-sampling`` (Default value = None)

  * ``None`` All training examples are considered for learning a new rule.
  * ``with-replacement`` The training examples to be considered for learning a new rule are selected randomly with replacement. The following options may be provided using the bracket notation:
  
    * ``sample_size`` (Default value = 1.0) The percentage of examples to be included in a sample, e.g., a value of 0.6 corresponds to 60% of the available examples. Must be in (0, 1).

  * ``without-replacement`` The training examples to be considered for learning a new rule are selected randomly without replacement. The following options may be provided using the bracket notation:
  
    * ``sample_size`` (Default value = 0.66) The percentage of examples to be included in a sample, e.g., a value of 0.6 corresponds to 60% of the available examples. Must be in (0, 1).

  * ``stratified-label-wise`` The training examples to be considered for learning a new rule are selected according to an iterative stratified sampling method that ensures that for each label the proportion of relevant and irrelevant examples is maintained. The following options may be provided using the bracket notation:
  
    * ``sample_size`` (Default value = 0.66) The percentage of examples to be included in a sample, e.g., a value of 0.6 corresponds to 60% of the available examples. Must be in (0, 1).

  * ``stratified-example-wise`` The training examples to be considered for learning a new rule are selected according to stratified sampling method, where distinct label vectors are treated as individual classes. The following options may be provided using the bracket notation:
  
    * ``sample_size`` (Default value = 0.66) The percentage of examples to be included in a sample, e.g., a value of 0.6 corresponds to 60% of the available examples. Must be in (0, 1).

* ``--recalculate-predictions`` (Default value = true)

  * ``true`` The predictions of rules are recalculated on the entire training data, if the parameter --instance-sampling is not set to None.
  * ``false`` The predictions of rules are not recalculated.

* ``--holdout`` (Default value = None)

  * ``None`` No holdout set is created.
  * ``random`` The available examples are randomly split into a training set and a holdout set. The following options may be provided using the bracket notation:
  
    * ``holdout_set_size`` (Default value = 0.33) The percentage of examples to be included in the holdout set, e.g., a value of 0.3 corresponds to 30% of the available examples. Must be in (0, 1).

  * ``stratified-label-wise`` The available examples are split into a training set and a holdout set according to an iterative stratified sampling method that ensures that for each label the proportion of relevant and irrelevant examples is maintained. The following options may be provided using the bracket notation:
  
    * ``holdout_set_size`` (Default value = 0.33) The percentage of examples to be included in the holdout set, e.g., a value of 0.3 corresponds to 30% of the available examples. Must be in (0, 1).

  * ``stratified-example-wise`` The available examples are split into a training set and a holdout set according to a stratified sampling method, where distinct label vectors are treated as individual classes. The following options may be provided using the bracket notation:
  
    * ``holdout_set_size`` (Default value = 0.33) The percentage of examples to be included in the holdout set, e.g., a value of 0.3 corresponds to 30% of the available examples. Must be in (0, 1).

* ``--early-stopping`` (Default value = None)

  * ``None`` No strategy for early-stopping is used.
  * ``loss`` Stops the induction of new rules as soon as the performance of the model does not improve on a holdout set, according to the loss function. This parameter does only have an effect if the parameter --holdout is set to a value greater than 0. The following options may be provided using the bracket notation:

    * ``min_rules`` (Default value = 100) The minimum number of rules. Must be at least 1.
    * ``update_interval`` (Default value = 1) The interval to be used to update the quality of the current model, e.g., a value of 5 means that the model quality is assessed every 5 rules. Must be at least 1.
    * ``stop_interval`` (Default value = 1) The interval to be used to decide whether the induction of rules should be stopped, e.g., a value of 10 means that the rule induction might be stopped after 10, 20, ... rules. Must be a multiple of update_interval.
    * ``num_past`` (Default value = 50) The number of quality scores of past iterations to be stored in a buffer. Must be at least 1.
    * ``num_recent`` (Default value = 50) The number of quality scores of the most recent iterations to be stored in a buffer. Must be at least 1.
    * ``aggregation`` (Default value = min) The name of the aggregation function that should be used to aggregate the scores in both buffers. Must be min, max or avg.
    * ``min_improvement`` (Default value = 0.005) The minimum improvement in percent that must be reached when comparing the aggregated scores in both buffers for the rule induction to be continued. Must be in [0, 1].
    * ``force_stop`` (Default value = ``true``) ``true``, if the induction of rules should be forced to be stopped, if the stopping criterion is met, ``false``, if the time of stopping should only be stored.

* ``--feature-binning`` (Default value = None)

  * ``None`` No feature binning is used.
  * ``equal-width`` Examples are assigned to bins, based on their feature values, according to the equal-width binning method. The following options may be provided using the bracket notation:
  
    * ``bin_ratio`` (Default value = 0.33) A percentage that specifies how many bins should be used, e.g., a value of 0.3 means that the number of bins should be set to 30% of the number of distinct values for a feature.
    * ``min_bins`` (Default value = 2) The minimum number of bins to be used. Must be at least 2.
    * ``max_bins`` (Default value = 0) The maximum number of bins to be used. Must be at least min_bins or 0, if the number of bins should not be restricted.

  * ``equal-frequency``. Examples are assigned to bins, based on their feature values, according to the equal-frequency binning method. The following options may be provided using the bracket notation:
  
    * ``bin_ratio`` (Default value = 0.33) A percentage that specifies how many bins should be used, e.g., a value of 0.3 means that the number of bins should be set to 30% of the number of distinct values for a feature.
    * ``min_bins`` (Default value = 2) The minimum number of bins to be used. Must be at least 2.
    * ``max_bins`` (Default value = 0) The maximum number of bins to be used. Must be at least min_bins or 0, if the number of bins should not be restricted.

* ``--label-binning`` (Default Value = auto)

  * ``None`` No label binning is used.
  * ``auto`` The most suitable strategy for label-binning is chosen automatically based on the loss function and the type of rule heads.
  * ``equal-width`` The labels for which a rule may predict are assigned to bins according to the equal-width binning method. The following options may be provided using the bracket notation:

    * ``bin_ratio`` (Default value = 0.04) A percentage that specifies how many bins should be used, e.g., a value of 0.04 means that number of bins should be set to 4% of the number of labels.
    * ``min_bins`` (Default value = 1) The minimum number of bins to be used. Must be at least 1.
    * ``max_bins`` (Default value = 0) The maximum number of bins to be used or 0, if the number of bins should not be restricted.

* ``--pruning`` (Default value = None)

  * ``None`` No pruning is used.
  * ``irep``. Subsequent conditions of rules may be pruned on a holdout set, similar to the IREP algorithm. Does only have an effect if the parameter --instance-sampling is not set to None.

* ``--min-coverage`` (Default value = 1)

  * The minimum number of training examples that must be covered by a rule. Must be at least 1.

* ``--max-conditions`` (Default value = 0)

  * The maximum number of conditions to be included in a rule's body. Must be at least 1 or 0, if the number of conditions should not be restricted.

* ``--max-head-refinements`` (Default value = 1)

  * The maximum number of times the head of a rule may be refined. Must be at least 1 or 0, if the number of refinements should not be restricted.

* ``--head-type`` (Default value = auto)

  * ``auto`` The most suitable type of rule heads is chosen automatically based on the loss function.
  * ``single-label`` If all rules should predict for a single label.
  * ``complete`` If all rules should predict for all labels simultaneously, potentially capturing dependencies between the labels.

* ``--shrinkage`` (Default value = 0.3)

  * The shrinkage parameter, a.k.a. the learning rate, to be used. Must be in (0, 1].

* ``--loss`` (Default value = logistic-label-wise)

  * ``logistic-label-wise`` A variant of the logistic loss function that is applied to each label individually.
  * ``logistic-example-wise`` A variant of the logistic loss function that takes all labels into account at the same time.
  * ``squared-error-label-wise`` A variant of the Squared error loss that is applied to each label individually.
  * ``hinge-label-wise`` A variant of the Hinge loss that is applied to each label individually.

* ``--predictor`` (Default value = auto)

  * ``auto`` The most suitable strategy for making predictions is chosen automatically, depending on the loss function.
  * ``label-wise`` The prediction for an example is determined for each label independently.
  * ``example-wise`` The label vector that is predicted for an example is chosen from the set of label vectors encountered in the training data.

* ``--l2-regularization-weight`` (Default value = 1.0)

  * The weight of the L2 regularization. Must be at least 0. If 0 is used, the L2 regularization is turned off entirely. Increasing the value causes the model to become more conservative.

**Multithreading**

The following parameters allow to enable multi-threading for different aspects of the algorithm:

* ``--parallel-rule-refinement`` (Default value = auto)

  * ``auto`` The number of threads to be used to search for potential refinements of rules in parallel is chosen automatically, depending on the loss function.
  * ``false`` No multi-threading is used to search for potential refinements of rules.
  * ``true`` Multi-threading is used to search for potential refinements of rules in parallel. The following options may be provided using the bracket notation:

    * ``num_threads`` (Default value = 0) The number of threads to be used. Must be at least 1 or 0, if the number of cores available on the machine should be used.

* ``--parallel-statistic-update`` (Default value = auto)

  * ``auto`` The number of threads to be used to calculate the gradients and Hessians for different examples in parallel is chosen automatically, depending on the loss function.
  * ``false`` No multi-threading is used to calculate the gradients and Hessians of different examples.
  * ``true`` Multi-threading is used to calculate the gradients and Hessians of different examples in parallel. The following options may be provided using the bracket notation:

    * ``num_threads`` (Default value = 0) The number of threads to be used. Must be at least 1 or 0, if the number of cores available on the machine should be used.

* ``--parallel-prediction`` (Default value = true)

  * ``false`` No multi-threading is used to obtain predictions for different examples.
  * ``true`` Multi-threading is used to obtain predictions for different examples in parallel. The following options may be provided using the bracket notation:

    * ``num_threads`` (Default value = 0) The number of threads to be used. Must be at least 1 or 0, if the number of cores available on the machine should be used.
