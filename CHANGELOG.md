# Changelog

### Version 0.4.0 (Mar 27, 2021)

A major update to the BOOMER algorithm that introduces the following changes:

* Large parts of the code have been refactored, and the core algorithm has been migrated to C++ entirely. It is now built and compiled using Meson and Ninja, which results in drastically reduced compile times.
* The (label- and example-wise) logistic loss functions have been rewritten to better prevent numerical problems.
* Approximate methods for evaluating potential conditions of rules, based on unsupervised binning methods (currently equal-width- and equal-frequency-binning), have been added.
* The parameter `--predictor` does now allow using different algorithms for prediction (`label-wise` or `example-wise`).
* An early stopping mechanism has been added, which allows to stop the induction of rules as soon as the quality of the model does not improve on a holdout set.    
* Multi-threading can be used to parallelize the prediction for different examples across multiple CPU cores.
* Multi-threading can be used to parallelize the calculation of gradients and Hessians for different examples across multiple CPU cores.
* Probability estimates can be predicted when using the loss function `label-wise-logistic-loss`.
* The algorithm does now support data sets with missing feature values.
* The loss function `label-wise-squared-hinge-loss` has been added. 
* Experiments using single-label data sets are now supported out of the box.

### Version 0.3.0 (Sep 14, 2020)

A major update to the BOOMER algorithm that features the following changes:

* Large parts of the code (loss functions, calculation of gradients/Hessians, calculation of predictions/quality scores) have been refactored and rewritten in C++. This comes with a constant speed-up of training times.
* Multi-threading can be used to parallelize the evaluation of a rule's potential refinements across multiple CPU cores.
* Sparse ground truth label matrices can now be used for training, which may reduce the memory footprint in case of large data sets.
* Additional parameters (`feature-format` and `label-format`) that allow to specify the preferred format of the feature and label matrix have been added.

### Version 0.2.0 (Jun 28, 2020)

A major update to the BOOMER algorithm that features the following changes:

* Includes many refactorings and quality of live improvements. Code that is not directly related with the algorithm, such as the implementation of baselines, has been removed.
* The algorithm is now able to natively handle nominal features without the need for pre-processing techniques such as one-hot encoding.
* Sparse feature matrices can now be used for training and prediction, which reduces the memory footprint and results in a significant speed-up of training times on some data sets.
* Additional hyper-parameters (`min_coverage`, `max_conditions` and `max_head_refinements`) that provide fine-grained control over the specificity/generality of rules have been added.

### Version 0.1.0 (Jun 22, 2020)

The first version of the BOOMER algorithm used in the following publication:

*Rapp M., Loza Mencía E., Fürnkranz J., Nguyen VL., Hüllermeier E. (2020) Learning Gradient Boosted Multi-label Classification Rules. In: Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2020. Lecture Notes in Computer Science, pp. 124-140, vol 12459. Springer, Cham*

This version supports the following features to learn an ensemble of boosted classification rules:

* Different label-wise or example-wise loss functions can be minimized during training (optionally using L2 regularization).
* The rules may predict for a single label, or for all labels (which enables to model local label dependencies).
* When learning a new rule, random samples of the training examples, features or labels may be used, including different techniques such as sampling with or without replacement.
* The impact of individual rules on the ensemble can be controlled using shrinkage.
* The conditions of a recently induced rule can be pruned based on a hold-out set.
* The algorithm currently only supports numerical or ordinal features. Nominal features can be handled by using one-hot encoding.
