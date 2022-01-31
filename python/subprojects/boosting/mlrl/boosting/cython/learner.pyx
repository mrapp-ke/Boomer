"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.boosting.cython.label_binning cimport EqualWidthLabelBinningConfig
from mlrl.boosting.cython.post_processor cimport ConstantShrinkageConfig
from mlrl.boosting.cython.regularization cimport ManualRegularizationConfig

from libcpp.memory cimport make_unique
from libcpp.utility cimport move

from scipy.linalg.cython_blas cimport ddot, dspmv
from scipy.linalg.cython_lapack cimport dsysv


cdef class BoostingRuleLearnerConfig(RuleLearnerConfig):
    """
    Allows to configure a rule learner that makes use of gradient boosting.
    """

    def __cinit__(self):
        self.rule_learner_config_ptr = createBoostingRuleLearnerConfig()

    cdef IRuleLearnerConfig* get_rule_learner_config_ptr(self):
        return self.rule_learner_config_ptr.get()

    def use_automatic_feature_binning(self):
        """
        Configures the rule learning to automatically decide whether a method for the assignment of numerical feature
        values to bins should be used or not.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAutomaticFeatureBinning()

    def use_constant_shrinkage_post_processor(self) -> ConstantShrinkageConfig:
        """
        Configures the rule learner to use a post-processor that shrinks the weights of rules by a constant "shrinkage"
        parameter.

        :return: A `ConstantShrinkageConfig` that allows further configuration of the post-processor
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IConstantShrinkageConfig* config_ptr = &rule_learner_config_ptr.useConstantShrinkagePostProcessor()
        cdef ConstantShrinkageConfig config = ConstantShrinkageConfig.__new__(ConstantShrinkageConfig)
        config.config_ptr = config_ptr
        return config

    def use_automatic_parallel_rule_refinement(self):
        """
        Configures the rule learner to automatically decide whether multi-threading should be used for the parallel
        refinement of rules or not.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAutomaticParallelRuleRefinement()

    def use_automatic_parallel_statistic_update(self):
        """
        Configures the rule learner to automatically decide whether multi-threading should be used for the parallel
        update of statistics or not.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAutomaticParallelStatisticUpdate()

    def use_automatic_heads(self):
        """
        Configures the rule learner to automatically decide for the type of rule heads to be used.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAutomaticHeads()

    def use_single_label_heads(self):
        """
        Configures the rule learner to induce rules with single-label heads that predict for a single label.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useSingleLabelHeads()

    def use_complete_heads(self):
        """
        Configures the rule learner to induce rules with complete heads that predict for all available labels.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useCompleteHeads()

    def use_no_l1_regularization(self):
        """
        Configures the rule learner to not use L1 regularization.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoL1Regularization()

    def use_l1_regularization(self) -> ManualRegularizationConfig:
        """
        Configures the rule learner to use L1 regularization.

        :return: A `ManualRegularizationConfig` that allows further configuration of the regularization term
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IManualRegularizationConfig* config_ptr = &rule_learner_config_ptr.useL1Regularization()
        cdef ManualRegularizationConfig config = ManualRegularizationConfig.__new__(ManualRegularizationConfig)
        config.config_ptr = config_ptr
        return config

    def use_no_l2_regularization(self):
        """
        Configures the rule learner to not use L2 regularization.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoL2Regularization()

    def use_l2_regularization(self) -> ManualRegularizationConfig:
        """
        Configures the rule learner to use L2 regularization.

        :return: A `ManualRegularizationConfig` that allows further configuration of the regularization term
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IManualRegularizationConfig* config_ptr = &rule_learner_config_ptr.useL2Regularization()
        cdef ManualRegularizationConfig config = ManualRegularizationConfig.__new__(ManualRegularizationConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_logistic_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the logistic loss
        that is applied example-wise.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useExampleWiseLogisticLoss()

    def use_label_wise_logistic_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the logistic loss
        that is applied label-wise.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLabelWiseLogisticLoss()

    def use_label_wise_squared_error_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared error
        loss that is applied label-wise.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLabelWiseSquaredErrorLoss()

    def use_label_wise_squared_hinge_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared hinge
        loss that is applied label-wise.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLabelWiseSquaredHingeLoss()

    def use_no_label_binning(self):
        """
        Configures the rule learner to not use any method for the assignment of labels to bins.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useNoLabelBinning()

    def use_automatic_label_binning(self):
        """
        Configures the rule learner to automatically decide whether a method for the assignment of labels to bins should
        be used or not.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useAutomaticLabelBinning()

    def use_equal_width_label_binning(self) -> EqualWidthLabelBinningConfig:
        """
        Configures the rule learner to use a method for the assignment of labels to bins in a way such that each bin
        contains labels for which the predicted score is expected to belong to the same value range.

        :return: A `EqualWidthLabelBinningConfig` that allows further configuration of the method for the assignment of
                 labels to bins
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        cdef IEqualWidthLabelBinningConfig* config_ptr = &rule_learner_config_ptr.useEqualWidthLabelBinning()
        cdef EqualWidthLabelBinningConfig config = EqualWidthLabelBinningConfig.__new__(EqualWidthLabelBinningConfig)
        config.config_ptr = config_ptr
        return config

    def use_example_wise_classification_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting whether individual labels are relevant or
        irrelevant by summing up the scores that are provided by an existing rule-based model and comparing the
        aggregated score vector to the known label vectors according to a certain distance measure. The label vector
        that is closest to the aggregated score vector is finally predicted.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useExampleWiseClassificationPredictor()

    def use_label_wise_classification_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting whether individual labels are relevant or
        irrelevant by summing up the scores that are provided by the individual rules of an existing rule-based model
        and transforming them into binary values according to a certain threshold that is applied to each label
        individually.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLabelWiseClassificationPredictor()

    def use_automatic_classification_predictor(self):
        """
        Configures the rule learner to automatically decide for a predictor for predicting whether individual labels are
        relevant or irrelevant.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLabelWiseClassificationPredictor()

    def use_label_wise_regression_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting regression scores by summing up the scores that
        are provided by the individual rules of an existing rule-based model for each label individually.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLabelWiseRegressionPredictor()

    def use_label_wise_probability_predictor(self):
        """
        Configures the rule learner to use a predictor for predicting probability estimates by summing up the scores
        that are provided by individual rules of an existing rule-based models and transforming the aggregated scores
        into probabilities according to a certain transformation function that is applied to each label individually.
        """
        cdef IBoostingRuleLearnerConfig* rule_learner_config_ptr = self.rule_learner_config_ptr.get()
        rule_learner_config_ptr.useLabelWiseProbabilityPredictor()


cdef class BoostingRuleLearner(RuleLearner):
    """
    A rule learner that makes use of gradient boosting.
    """

    def __cinit__(self, BoostingRuleLearnerConfig config not None):
        """
        :param config: The configuration that should be used by the rule learner
        """
        self.rule_learner_ptr = createBoostingRuleLearner(move(config.rule_learner_config_ptr), ddot, dspmv, dsysv)

    cdef IRuleLearner* get_rule_learner_ptr(self):
        return self.rule_learner_ptr.get()
