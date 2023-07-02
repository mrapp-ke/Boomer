.. _parameters:

Parameters
----------

The behavior of the BOOMER algorithm can be controlled in a fine-grained manner via a large number of parameters. Values for these parameters may be provided as constructor arguments to the class ``Boomer`` as shown in the section :ref:`usage`. They can also be used to configure the algorithm when using the :ref:`testbed`.

All of the parameters that are mentioned below are optional. If not specified manually, default settings that work well in most of the cases are used. In the following, an overview of all available parameters, as well as their default values, is provided.

.. note::
    Each parameter is identified by an unique name. Depending on the type of a parameter, it either accepts numbers as possible values or allows to specify a string that corresponds to a predefined set of possible values (boolean values are also represented as strings).

    In addition to the specified value, some parameters allow to provide additional options as key-value pairs. These options must be provided by using the following bracket notation:

    ``'value{key1=value1,key2=value2}'``

    For example, the parameter ``feature_binning`` allows to provide additional options and may be configured as follows:

    ``'equal-width{bin_ratio=0.33,min_bins=2,max_bins=64}'``

**Data Format**

The following parameters allow to specify the preferred format for the representation of the training data:

* ``feature_format`` (Default value = ``'auto'``)

  * ``'auto'`` The most suitable format for representation of the feature matrix is chosen automatically by estimating which representation requires less memory.
  * ``'dense'`` Enforces that the feature matrix is stored using a dense format.
  * ``'sparse'`` Enforces that the feature matrix is stored using a sparse format. Using a sparse format may reduce the memory footprint and/or speed up the training process on some data sets.

* ``label_format`` (Default value = ``'auto'``)

  * ``'auto'`` The most suitable format for representation of the label matrix is chosen automatically by estimating which representation requires less memory.
  * ``'dense'`` Enforces that the label matrix is stored using a dense format.
  * ``'sparse'`` Enforces that the label matrix is stored using a sparse format. Using a sparse format may reduce the memory footprint on some data sets.

* ``prediction_format`` (Default value = ``'auto'``)

  * ``'auto'`` The most suitable format for the representation of predictions is chosen automatically based on the sparsity of the ground truth labels supplied for training.
  * ``'dense'`` Enforces that predictions are stored using a dense format.
  * ``'sparse'`` Enforces that predictions are stored using a sparse format, if supported. Using a sparse format may reduce the memory footprint on some data sets.

**Algorithmic Parameters**

The following parameters allow to control the behavior of the algorithm:

* ``random_state`` (Default value = ``1``)

  * The seed to be used by random number generators. Must be at least 1.

* ``default_rule`` (Default value = ``'auto'``)

  * ``'auto'`` A default rule that provides a default prediction for all examples is included as the first rule of a model unless it prevents a sparse format for the representation of gradients and Hessians from being used (see parameter ``statistic_format``).
  * ``'true'`` A default rule that provides a default prediction for all examples is included as the first rule of a model.
  * ``'false'`` No default rule is used.

* ``rule_induction`` (Default value = ``'top-down-greedy'``)

  * ``'top-down-greedy'`` A greedy top-down search, where rules are successively refined by adding new conditions, is used for the induction of individual rules. The following options may be provided using the bracket notation:

    * ``max_conditions`` (Default value = ``0``) The maximum number of conditions to be included in a rule's body. Must be at least 1 or 0, if the number of conditions should not be restricted.
    * ``min_coverage`` (Default value = ``1``) The minimum number of training examples that must be covered by a rule. Must be at least 1.
    * ``min_support`` (Default value = ``0.0``) The minimum support, i.e., the fraction of training examples that must be covered by a rule. Must be in [0, 1] or 0, if the support of rules should not be restricted.
    * ``max_head_refinements`` (Default value = ``1``) The maximum number of times the head of a rule may be refined. Must be at least 1 or 0, if the number of refinements should not be restricted.
    * ``recalculate_predictions`` (Default value = ``'true'``) ``'true'``, if the predictions of rules should be recalculated on the entire training data if the parameter ``instance_sampling`` is not set to ``'none'``, ``'false'``, if the predictions of rules should not be recalculated.

  * ``'top-down-beam-search'`` A top-down beam search, where rules are successively refined by adding new conditions, is used for the induction of individual rules. The following options may be provided using the bracket notation:

    * ``beam_width`` (Default value = ``4``) The width to be used by the beam search. A larger value tends to result in more accurate rules being found, but negatively affects the training time. Must be at least 2
    * ``resample_features`` (Default value = ``'false'``) ``'true'``, if a new sample of the available features should be created for each rule that is refined during a beam search, ``'false'`` otherwise. Does only have an effect if the parameter ``feature_sampling`` is not set to ``'none'``.
    * ``max_conditions`` (Default value = ``0``) The maximum number of conditions to be included in a rule's body. Must be at least 2 or 0, if the number of conditions should not be restricted.
    * ``min_coverage`` (Default value = ``1``) The minimum number of training examples that must be covered by a rule. Must be at least 1.
    * ``min_support`` (Default value = ``0.0``) The minimum support, i.e., the fraction of training examples that must be covered by a rule. Must be in [0, 1] or 0, if the support of rules should not be restricted.
    * ``max_head_refinements`` (Default value = ``1``) The maximum number of times the head of a rule may be refined. Must be at least 1 or 0, if the number of refinements should not be restricted.
    * ``recalculate_predictions`` (Default value = ``'true'``) ``'true'``, if the predictions of rules should be recalculated on the entire training data if the parameter ``instance_sampling`` is not set to ``'none'``, ``'false'``, if the predictions of rules should not be recalculated.

* ``max_rules`` (Default value = ``1000``)

  * The maximum number of rules to be learned (including the default rule). Must be at least 1 or 0, if the number of rules should not be restricted.

* ``time_limit`` (Default value = ``0``)

  * The duration in seconds after which the induction of rules should be canceled. Must be at least 1 or 0, if no time limit should be set.

* ``label_sampling`` (Default value = ``'none'``)

  * ``'none'`` All labels are considered for learning a new rule.
  * ``'round-robin'`` A single label to be considered when learning a new rule is chosen in a round-robin fashion, i.e., the first rule is concerned with the first label, the second one with the second label, and so on. When the last label was reached, the procedure restarts at the first label.
  * ``'without-replacement'`` The labels to be considered when learning a new rule are chosen randomly. The following options may be provided using the bracket notation:
  
    * ``num_samples`` (Default value = ``1``) The number of labels the be included in a sample. Must be at least 1.

* ``feature_sampling`` (Default value = ``'without-replacement'``)

  * ``'none'`` All features are considered for learning a new rule.
  * ``'without-replacement'`` A random subset of the features is used to search for the refinements of rules. The following options may be provided using the bracket notation:

    * ``sample_size`` (Default value = ``0``) The percentage of features to be included in a sample. For example, a value of 0.6 corresponds to 60% of the features. Must be in (0, 1] or 0, if the sample size should be calculated as log2(A - 1) + 1), where A denotes the number of available features.
    * ``num_retained`` (Default value = ``0``) The number of trailing features to be always included in a sample. For example, a value of 2 means that the last two features are always retained.

* ``instance_sampling`` (Default value = ``'none'``)

  * ``'none'`` All training examples are considered for learning a new rule.
  * ``'with-replacement'`` The training examples to be considered for learning a new rule are selected randomly with replacement. The following options may be provided using the bracket notation:
  
    * ``sample_size`` (Default value = ``1.0``) The percentage of examples to be included in a sample. For example, a value of 0.6 corresponds to 60% of the available examples. Must be in (0, 1).

  * ``'without-replacement'`` The training examples to be considered for learning a new rule are selected randomly without replacement. The following options may be provided using the bracket notation:
  
    * ``sample_size`` (Default value = ``0.66``) The percentage of examples to be included in a sample. For example, a value of 0.6 corresponds to 60% of the available examples. Must be in (0, 1).

  * ``'stratified-label-wise'`` The training examples to be considered for learning a new rule are selected according to an iterative stratified sampling method that ensures that for each label the proportion of relevant and irrelevant examples is maintained. The following options may be provided using the bracket notation:
  
    * ``sample_size`` (Default value = ``0.66``) The percentage of examples to be included in a sample. For example, a value of 0.6 corresponds to 60% of the available examples. Must be in (0, 1).

  * ``'stratified-example-wise'`` The training examples to be considered for learning a new rule are selected according to stratified sampling method, where distinct label vectors are treated as individual classes. The following options may be provided using the bracket notation:
  
    * ``sample_size`` (Default value = ``0.66``) The percentage of examples to be included in a sample. For example, a value of 0.6 corresponds to 60% of the available examples. Must be in (0, 1).

* ``holdout`` (Default value = ``'auto'``)

  * ``'none'`` No holdout set is created.
  * ``'auto'`` The most suitable strategy for creating a holdout set is chosen automatically, depending on whether a holdout set is needed according to the parameters ``--global_pruning``, ``--marginal-probability-calibration`` or ``--joint-probability-calibration``.
  * ``'random'`` The available examples are randomly split into a training set and a holdout set. The following options may be provided using the bracket notation:
  
    * ``holdout_set_size`` (Default value = ``0.33``) The percentage of examples to be included in the holdout set. For example, a value of 0.3 corresponds to 30% of the available examples. Must be in (0, 1).

  * ``'stratified-label-wise'`` The available examples are split into a training set and a holdout set according to an iterative stratified sampling method that ensures that for each label the proportion of relevant and irrelevant examples is maintained. The following options may be provided using the bracket notation:
  
    * ``holdout_set_size`` (Default value = ``0.33``) The percentage of examples to be included in the holdout set. For example, a value of 0.3 corresponds to 30% of the available examples. Must be in (0, 1).

  * ``'stratified-example-wise'`` The available examples are split into a training set and a holdout set according to a stratified sampling method, where distinct label vectors are treated as individual classes. The following options may be provided using the bracket notation:
  
    * ``holdout_set_size`` (Default value = ``0.33``) The percentage of examples to be included in the holdout set. For example, a value of 0.3 corresponds to 30% of the available examples. Must be in (0, 1).

* ``global_pruning`` (Default value = ``'none'``)

  * ``'none'`` No strategy for pruning entire rules is used.
  * ``'post-pruning'`` Keeps track of the number of rules in a model that perform best on the training or holdout set according to the loss function. The following options may be provided using the bracket notation:

    * ``use_holdout_set`` (Default value = ``'true'``) ``'true'``, if the quality of the current model should be measured on the holdout set, if available, ``'false'``, if the training set should be used instead.
    * ``remove_unused_rules`` (Default value = ``'true'``) ``'true'``, if unused rules should be removed from the final model, ``'false'`` otherwise.
    * ``min_rules`` (Default value = ``100``) The minimum number of rules that must be included in a model. Must be at least 1
    * ``interval`` (Default value = ``1``) The interval to be used to check whether the current model is the best one evaluated so far. For example, a value of 10 means that the best model may contain 10, 20, ... rules. Must be at least 1

  * ``'pre-pruning'`` Stops the induction of new rules as soon as the performance of the model does not improve on the training or holdout set according to the loss function. The following options may be provided using the bracket notation:

    * ``use_holdout_set`` (Default value = ``'true'``) ``'true'``, if the quality of the current model should be measured on the holdout set, if available, ``'false'``, if the training set should be used instead.
    * ``remove_unused_rules`` (Default value = ``'true'``) ``'true'``, if the induction of rules should be stopped as soon as the stopping criterion is met, ``'false'``, if additional rules should be included in the model without being used for prediction.
    * ``min_rules`` (Default value = ``100``) The minimum number of rules that must be included in a model. Must be at least 1.
    * ``update_interval`` (Default value = ``1``) The interval to be used to update the quality of the current model. For example, a value of 5 means that the model quality is assessed every 5 rules. Must be at least 1.
    * ``stop_interval`` (Default value = ``1``) The interval to be used to decide whether the induction of rules should be stopped. For example, a value of 10 means that the rule induction might be stopped after 10, 20, ... rules. Must be a multiple of update_interval.
    * ``num_past`` (Default value = ``50``) The number of quality scores of past iterations to be stored in a buffer. Must be at least 1.
    * ``num_recent`` (Default value = ``50``) The number of quality scores of the most recent iterations to be stored in a buffer. Must be at least 1.
    * ``aggregation`` (Default value = ``'min'``) The name of the aggregation function that should be used to aggregate the scores in both buffers. Must be ``'min'``, ``'max'`` or ``'avg'``.
    * ``min_improvement`` (Default value = ``0.005``) The minimum improvement in percent that must be reached when comparing the aggregated scores in both buffers for the rule induction to be continued. Must be in [0, 1].

* ``rule_pruning`` (Default value = ``'none'``)

  * ``'none'`` No method for pruning individual rules is used.
  * ``'irep'`` Trailing conditions of rules may be pruned on a holdout set, similar to the IREP algorithm. Does only have an effect if the parameter ``instance_sampling`` is not set to ``'none'``.

* ``sequential_post_optimization`` (Default value = ``'false'``)

    * ``'false'`` Sequential post-optimization is not used.
    * ``'true'`` Each rule in a previously learned model is optimized by being relearned in the context of the other rules. The following options may be provided using the bracket notation:

      * ``num_iterations`` (Default value = ``2``) The number of times each rule should be relearned. Must be at least 1.
      * ``refine_heads`` (Default value = ``'false'``) ``'true'``, if the heads of rules may be refined when being relearned, ``'false'``, if the relearned rules should be predict for the same labels as the original rules.
      * ``resample_features`` (Default value = ``'true'``) ``'true'``, if a new sample of the available features should be created whenever a new rule is refined, ``'false'``, if the conditions of the new rule should use the same features as the original rule

* ``feature_binning`` (Default value = ``'none'``)

  * ``'none'`` No feature binning is used.
  * ``'equal-width'`` Examples are assigned to bins, based on their feature values, according to the equal-width binning method. The following options may be provided using the bracket notation:
  
    * ``bin_ratio`` (Default value = ``0.33``) A percentage that specifies how many bins should be used. For example, a value of 0.3 means that the number of bins should be set to 30% of the number of distinct values for a feature.
    * ``min_bins`` (Default value = ``2``) The minimum number of bins. Must be at least 2.
    * ``max_bins`` (Default value = ``0``) The maximum number of bins. Must be at least min_bins or 0, if the number of bins should not be restricted.

  * ``'equal-frequency'``. Examples are assigned to bins, based on their feature values, according to the equal-frequency binning method. The following options may be provided using the bracket notation:
  
    * ``bin_ratio`` (Default value = ``0.33``) A percentage that specifies how many bins should be used. For example, a value of 0.3 means that the number of bins should be set to 30% of the number of distinct values for a feature.
    * ``min_bins`` (Default value = ``2``) The minimum number of bins. Must be at least 2.
    * ``max_bins`` (Default value = ``0``) The maximum number of bins. Must be at least min_bins or 0, if the number of bins should not be restricted.

* ``label_binning`` (Default Value = ``'auto'``)

  * ``'none'`` No label binning is used.
  * ``'auto'`` The most suitable strategy for label-binning is chosen automatically based on the loss function and the type of rule heads.
  * ``'equal-width'`` The labels for which a rule may predict are assigned to bins according to the equal-width binning method. The following options may be provided using the bracket notation:

    * ``bin_ratio`` (Default value = ``0.04``) A percentage that specifies how many bins should be used. For example, a value of 0.04 means that number of bins should be set to 4% of the number of labels.
    * ``min_bins`` (Default value = ``1``) The minimum number of bins. Must be at least 1.
    * ``max_bins`` (Default value = ``0``) The maximum number of bins or 0, if the number of bins should not be restricted.

* ``head_type`` (Default value = ``'auto'``)

  * ``'auto'`` The most suitable type of rule heads is chosen automatically, depending on the loss function.
  * ``'single-label'`` If all rules should predict for a single label.
  * ``'partial-fixed'`` If all rules should predict for a predefined number of labels. The following options may be provided using the bracket notation:

    * ``label_ratio`` (Default value = ``0.0``) A percentage that specifies for how many labels the rules should predict or 0, if the percentage should be calculated based on the average label cardinality. For example, a value of 0.05 means that the rules should predict for 5% of the available labels.
    * ``min_labels`` (Default value = ``2``) The minimum number of labels for which the rules should predict. Must be at least 2.
    * ``max_labels`` (Default value = ``0``) The maximum number of labels for which the rules should predict or 0, if the number of predictions should not be restricted.

  * ``'partial-dynamic'`` If all rules should predict for a subset of the available labels that is determined dynamically. The following options may be provided using the bracket notation:

    * ``threshold`` (Default value = ``0.02``) A threshold that affects for how many labels the rules should predict. A smaller threshold results in less labels being selected. A greater threshold results in more labels being selected. E.g., a threshold of 0.02 means that a rule will only predict for a label if the estimated predictive quality ``q`` for this particular label satisfies the inequality ``q^exponent > q_best^exponent * (1 - 0.02)``, where ``q_best`` is the best quality among all labels. Must be in (0, 1)
    * ``exponent`` (Default value = ``2.0``) An exponent that is used to weigh the estimated predictive quality for individual labels. E.g., an exponent of 2 means that the estimated predictive quality `q` for a particular label is weighed as ``q^2``. Must be at least 1.

  * ``'complete'`` If all rules should predict for all labels simultaneously, potentially capturing dependencies between the labels.

* ``statistic_format`` (Default value ``'auto'``)

  * ``'auto'`` The most suitable format for the representation of gradients and Hessians is chosen automatically, depending on the loss function, the type of rule heads, the characteristics of the label matrix and whether a default rule is used or not.
  * ``'dense'`` A dense format is used for the representation of gradients and Hessians.
  * ``'sparse'`` A sparse format is used for the representation of gradients and Hessians, if supported by the loss function.

* ``shrinkage`` (Default value = ``0.3``)

  * The shrinkage parameter, a.k.a. the "learning rate", that is used to shrink the weight of individual rules. Must be in (0, 1].

* ``loss`` (Default value = ``'logistic-label-wise'``)

  * ``'logistic-label-wise'`` A variant of the logistic loss function that is applied to each label individually.
  * ``'logistic-example-wise'`` A variant of the logistic loss function that takes all labels into account at the same time.
  * ``'squared-error-label-wise'`` A variant of the squared error loss that is applied to each label individually.
  * ``'squared-error-example-wise'`` A variant of the squared error loss that takes all labels into account at the same time.
  * ``'squared-hinge-label-wise'`` A variant of the squared hinge loss that is applied to each label individually.
  * ``'squared-hinge-example-wise'`` A variant fot he squared hinge loss that takes all labels into account at the same time.

* ``marginal_probability_calibration`` (Default value = ``'none'``)

  * ``'none'`` Marginal probabilities are not calibrated.
  * ``'isotonic'`` Marginal probabilities are calibrated via isotonic regression.

    * ``'use_holdout_set'`` (Default value = ``'true'``) ``'true'``, if the calibration model should be fit to the examples in the holdout set, if available, ``'false'``, if the training set should be used instead.

* ``joint_probability_calibration`` (Default value = ``'none'``)

  * ``'none'`` Joint probabilities are not calibrated.
  * ``'isotonic'`` Joint probabilities are calibrated via isotonic regression.

    * ``'use_holdout_set'`` (Default value = ``'true'``) ``'true'``, if the calibration model should be fit to the examples in the holdout set, if available, ``'false'``, if the training set should be used instead.

* ``binary_predictor`` (Default value = ``'auto'``)

  * ``'auto'`` The most suitable strategy for predicting binary labels is chosen automatically, depending on the loss function.
  * ``'label-wise'`` The prediction for an example is determined for each label independently. The following options may be provided using the bracket notation:

    * ``based_on_probabilities`` (Default value = ``'false'``) ``'true'``, if binary predictions should be derived from probability estimates rather than regression scores if supported by the loss function, ``'false'`` otherwise.
    * ``use_probability_calibration`` (Default value = ``'true'``) ``'true'``, if a model for the calibration of probabilities should be used, if available, ``'false'`` otherwise. Does only have an effect if the option ``based_on_probabilities`` is set to ``'true'``.

  * ``'example-wise'`` The label vector that is predicted for an example is chosen from the set of label vectors encountered in the training data. The following options may be provided using the bracket notation:

    * ``based_on_probabilities`` (Default value = ``'false'``) ``'true'``, if binary predictions should be derived from probability estimates rather than regression scores if supported by the loss function, ``'false'`` otherwise.
    * ``use_probability_calibration`` (Default value = ``'true'``) ``'true'``, if a model for the calibration of probabilities should be used, if available, ``'false'`` otherwise. Does only have an effect if the option ``based_on_probabilities`` is set to ``'true'``.

  * ``'gfm'`` The label vector that is predicted for an example is chosen according to the general F-measure maximizer (GFM).

    * ``use_probability_calibration`` (Default value = ``'true'``) ``'true'``, if a model for the calibration of probabilities should be used, if available, ``'false'`` otherwise.

* ``probability_predictor`` (Default value = ``'auto'``)

  * ``'auto'`` The most suitable strategy for predicting probability estimates is chosen automatically, depending on the loss function.
  * ``'label-wise'`` The prediction for an example is determined for each label independently

    * ``use_probability_calibration`` (Default value = ``'true'``) ``'true'``, if a model for the calibration of probabilities should be used, if available, ``'false'`` otherwise.

  * ``'marginalized'`` The prediction for an example is determined via marginalization over the set of label vectors encountered in the training data.

    * ``use_probability_calibration`` (Default value = ``'true'``) ``'true'``, if a model for the calibration of probabilities should be used, if available, ``'false'`` otherwise.

* ``l1_regularization_weight`` (Default value = ``0.0``)

  * The weight of the L1 regularization. Must be at least 0. If 0 is used, the L1 regularization is turned off entirely. Increasing the value causes the model to become more conservative.

* ``l2_regularization_weight`` (Default value = ``1.0``)

  * The weight of the L2 regularization. Must be at least 0. If 0 is used, the L2 regularization is turned off entirely. Increasing the value causes the model to become more conservative.

**Multi-Threading**

The following parameters allow to specify whether multi-threading should be used for different aspects of the algorithm:

* ``parallel_rule_refinement`` (Default value = ``'auto'``)

  * ``'auto'`` The number of threads to be used to search for potential refinements of rules in parallel is chosen automatically, depending on the loss function.
  * ``'false'`` No multi-threading is used to search for potential refinements of rules.
  * ``'true'`` Multi-threading is used to search for potential refinements of rules in parallel. The following options may be provided using the bracket notation:

    * ``num_threads`` (Default value = ``0``) The number of threads to be used. Must be at least 1 or 0, if the number of cores available on the machine should be used.

* ``parallel_statistic_update`` (Default value = ``'auto'``)

  * ``'auto'`` The number of threads to be used to calculate the gradients and Hessians for different examples in parallel is chosen automatically, depending on the loss function.
  * ``'false'`` No multi-threading is used to calculate the gradients and Hessians of different examples.
  * ``'true'`` Multi-threading is used to calculate the gradients and Hessians of different examples in parallel. The following options may be provided using the bracket notation:

    * ``num_threads`` (Default value = ``0``) The number of threads to be used. Must be at least 1 or 0, if the number of cores available on the machine should be used.

* ``parallel_prediction`` (Default value = ``'true'``)

  * ``'false'`` No multi-threading is used to obtain predictions for different examples.
  * ``'true'`` Multi-threading is used to obtain predictions for different examples in parallel. The following options may be provided using the bracket notation:

    * ``num_threads`` (Default value = ``0``) The number of threads to be used. Must be at least 1 or 0, if the number of cores available on the machine should be used.
