"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from abc import ABC, abstractmethod

from mlrl.boosting.cython.head_type import DynamicPartialHeadConfig, FixedPartialHeadConfig
from mlrl.boosting.cython.label_binning import EqualWidthLabelBinningConfig
from mlrl.boosting.cython.post_processor import ConstantShrinkageConfig
from mlrl.boosting.cython.prediction import ExampleWiseBinaryPredictorConfig, GfmBinaryPredictorConfig, \
    LabelWiseBinaryPredictorConfig, LabelWiseProbabilityPredictorConfig, MarginalizedProbabilityPredictorConfig
from mlrl.boosting.cython.probability_calibration import IsotonicJointProbabilityCalibratorConfig, \
    IsotonicMarginalProbabilityCalibratorConfig
from mlrl.boosting.cython.regularization import ManualRegularizationConfig


class AutomaticPartitionSamplingMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide whether a holdout set should be used or not.
    """

    @abstractmethod
    def use_automatic_partition_sampling(self):
        """
        Configures the rule learner to automatically decide whether a holdout set should be used or not.
        """
        pass

             
class AutomaticFeatureBinningMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide whether a method for the assignment of numerical feature
    values to bins should be used or not.
    """

    @abstractmethod
    def use_automatic_feature_binning(self):
        """
        Configures the rule learning to automatically decide whether a method for the assignment of numerical feature
        values to bins should be used or not.
        """
        pass
             
             
class AutomaticParallelRuleRefinementMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide whether multi-threading should be used for the parallel
    refinement of rules or not.
    """

    @abstractmethod
    def use_automatic_parallel_rule_refinement(self):
        """
        Configures the rule learner to automatically decide whether multi-threading should be used for the parallel
        refinement of rules or not.
        """
        pass
             
             
class AutomaticParallelStatisticUpdateMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide whether multi-threading should be used for the parallel
    update of statistics or not.
    """

    @abstractmethod
    def use_automatic_parallel_statistic_update(self):
        """
        Configures the rule learner to automatically decide whether multi-threading should be used for the parallel
        update of statistics or not.
        """
        pass
             
             
class ConstantShrinkageMixin(ABC):
    """
    Allows to configure a rule learner to use a post processor that shrinks the weights fo rules by a constant
    "shrinkage" parameter.
    """

    @abstractmethod
    def use_constant_shrinkage_post_processor(self) -> ConstantShrinkageConfig:
        """
        Configures the rule learner to use a post-processor that shrinks the weights of rules by a constant "shrinkage"
        parameter.

        :return: A `ConstantShrinkageConfig` that allows further configuration of the post-processor
        """
        pass
             
             
class NoL1RegularizationMixin(ABC):
    """
    Allows to configure a rule learner to not use L1 regularization.
    """

    @abstractmethod
    def use_no_l1_regularization(self):
        """
        Configures the rule learner to not use L1 regularization.
        """
        pass


class L1RegularizationMixin(ABC):
    """
    Allows to configure a rule learner to use L1 regularization.
    """

    @abstractmethod
    def use_l1_regularization(self) -> ManualRegularizationConfig:
        """
        Configures the rule learner to use L1 regularization.

        :return: A `ManualRegularizationConfig` that allows further configuration of the regularization term
        """
        pass
            
             
class NoL2RegularizationMixin(ABC):
    """
    Allows to configure a rule learner to not use L2 regularization.
    """

    @abstractmethod
    def use_no_l2_regularization(self):
        """
        Configures the rule learner to not use L2 regularization.
        """
        pass
            
             
class L2RegularizationMixin(ABC):
    """
    Allows to configure a rule learner to use L2 regularization.
    """

    @abstractmethod
    def use_l2_regularization(self) -> ManualRegularizationConfig:
        """
        Configures the rule learner to use L2 regularization.

        :return: A `ManualRegularizationConfig` that allows further configuration of the regularization term
        """
        pass
             
             
class NoDefaultRuleMixin(ABC):
    """
    Allows to configure a rule learner to not induce a default rule.
    """

    @abstractmethod
    def use_no_default_rule(self):
        """
        Configures the rule learner to not induce a default rule.
        """
        pass
             
             
class AutomaticDefaultRuleMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide whether a default rule should be induced or not.
    """

    @abstractmethod
    def use_automatic_default_rule(self):
        """
        Configures the rule learner to automatically decide whether a default rule should be induced or not.
        """
        pass
            
             
class CompleteHeadMixin(ABC):
    """
    Allows to configure a rule learner to induce rules with complete heads that predict for all available labels.
    """

    @abstractmethod
    def use_complete_heads(self):
        """
        Configures the rule learner to induce rules with complete heads that predict for all available labels.
        """
        pass
            
             
class FixedPartialHeadMixin(ABC):
    """
    Allows to configure a rule learner to induce rules with partial heads that predict for a predefined number of
    labels.
    """

    @abstractmethod
    def use_fixed_partial_heads(self) -> FixedPartialHeadConfig:
        """
        Configures the rule learner to induce rules with partial heads that predict for a predefined number of labels.

        :return: A `FixedPartialHeadConfig` that allows further configuration of the rule heads
        """
        pass
            
             
class DynamicPartialHeadMixin(ABC):
    """
    Allows to configure a rule learner to induce rules with partial heads that predict for a subset of the available
    labels that is determined dynamically.
    """

    @abstractmethod
    def use_dynamic_partial_heads(self) -> DynamicPartialHeadConfig:
        """
        Configures the rule learner to induce rules with partial heads that predict for a subset of the available labels
        that is determined dynamically. Only those labels for which the square of the predictive quality exceeds a
        certain threshold are included in a rule head.

        :return: A `DynamicPartialHeadConfig` that allows further configuration of the rule heads
        """
        pass
            
             
class SingleLabelHeadMixin(ABC):
    """
    Allows to configure a rule learner to induce rules with single-label heads that predict for a single label.
    """

    @abstractmethod
    def use_single_label_heads(self):
        """
        Configures the rule learner to induce rules with single-label heads that predict for a single label.
        """
        pass
            
             
class AutomaticHeadMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide for the type of rule heads that should be used.
    """

    @abstractmethod
    def use_automatic_heads(self):
        """
        Configures the rule learner to automatically decide for the type of rule heads to be used.
        """
        pass
            
             
class DenseStatisticsMixin(ABC):
    """
    Allows to configure a rule learner to use a dense representation of gradients and Hessians.
    """

    @abstractmethod
    def use_dense_statistics(self):
        """
        Configures the rule learner to use a dense representation of gradients and Hessians.
        """
        pass
            
             
class SparseStatisticsMixin(ABC):
    """
    Allows to configure a rule learner to use a sparse representation of gradients and Hessians, if possible.
    """

    @abstractmethod
    def use_sparse_statistics(self):
        """
        Configures the rule learner to use a sparse representation of gradients and Hessians, if possible.
        """
        pass
            
             
class AutomaticStatisticsMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide whether a dense or sparse representation of gradients and
    Hessians should be used.
    """

    @abstractmethod
    def use_automatic_statistics(self):
        """
        Configures the rule learner to automatically decide whether a dense or sparse representation of gradients and
        Hessians should be used.
        """
        pass
            
             
class ExampleWiseLogisticLossMixin(ABC):
    """
    Allows to configure a rule learner to use a loss function that implements a multi-label variant of the logistic loss
    that is applied example-wise.
    """

    @abstractmethod
    def use_example_wise_logistic_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the logistic loss
        that is applied example-wise.
        """
        pass
            
             
class ExampleWiseSquaredErrorLossMixin(ABC):
    """
    Allows to configure a rule learner to use a loss function that implements a multi-label variant of the squared error
    loss that is applied example-wise.
    """

    @abstractmethod
    def use_example_wise_squared_error_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared error
        loss that is applied example-wise.
        """
        pass
            
             
class ExampleWiseSquaredHingeLossMixin(ABC):
    """
    Allows to configure a rule learner to use a loss function that implements a multi-label variant of the squared hinge
    loss that is applied example-wise.
    """

    @abstractmethod
    def use_example_wise_squared_hinge_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared hinge
        loss that is applied example-wise.
        """
        pass
            
             
class LabelWiseLogisticLossMixin(ABC):
    """
    Allows to configure a rule learner to use a loss function that implements a multi-label variant of the logistic loss
    that is applied label-wise.
    """

    @abstractmethod
    def use_label_wise_logistic_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the logistic loss
        that is applied label-wise.
        """
        pass
            
             
class LabelWiseSquaredErrorLossMixin(ABC):
    """
    Allows to configure a rule learner to use a loss function that implements a multi-label variant of the squared error
    loss that is applied label-wise.
    """

    @abstractmethod
    def use_label_wise_squared_error_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared error
        loss that is applied label-wise.
        """
        pass
            
             
class LabelWiseSquaredHingeLossMixin(ABC):
    """
    Allows to configure a rule learner to use a loss function that implements a multi-label variant of the squared hinge
    loss that is applied label-wise.
    """

    @abstractmethod
    def use_label_wise_squared_hinge_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multi-label variant of the squared hinge
        loss that is applied label-wise.
        """
        pass
            
            
class NoLabelBinningMixin(ABC):
    """
    Allows to configure a rule learner to not use any method for the assignment of labels to bins.
    """

    @abstractmethod
    def use_no_label_binning(self):
        """
        Configures the rule learner to not use any method for the assignment of labels to bins.
        """
        pass
            
             
class EqualWidthLabelBinningMixin(ABC):
    """
    Allows to configure a rule learner to use a method for the assignment of labels to bins.
    """

    @abstractmethod
    def use_equal_width_label_binning(self) -> EqualWidthLabelBinningConfig:
        """
        Configures the rule learner to use a method for the assignment of labels to bins in a way such that each bin
        contains labels for which the predicted score is expected to belong to the same value range.

        :return: A `EqualWidthLabelBinningConfig` that allows further configuration of the method for the assignment of
                 labels to bins
        """
        pass
            
             
class AutomaticLabelBinningMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide whether a method for the assignment of labels to bins
    should be used or not.
    """

    @abstractmethod
    def use_automatic_label_binning(self):
        """
        Configures the rule learner to automatically decide whether a method for the assignment of labels to bins should
        be used or not.
        """
        pass
            
             
class IsotonicMarginalProbabilityCalibrationMixin(ABC):
    """
    Allows to configure a rule learner to calibrate marginal probabilities via isotonic regression.
    """

    @abstractmethod
    def use_isotonic_marginal_probability_calibration(self) -> IsotonicMarginalProbabilityCalibratorConfig:
        """
        Configures the rule learner to calibrate marginal probabilities via isotonic regression.

        :return: An `IsotonicMarginalProbabilityCalibratorConfig` that allows further configuration of the calibrator
        """
        pass


class IsotonicJointProbabilityCalibrationMixin(ABC):
    """
    Allows to configure a rule learner to calibrate joint probabilities via isotonic regression.
    """

    @abstractmethod
    def use_isotonic_joint_probability_calibration(self) -> IsotonicJointProbabilityCalibratorConfig:
        """
        Configures the rule learner to calibrate joint probabilities via isotonic regression.

        :return: An `IsotonicJointProbabilityCalibratorConfig` that allows further configuration of the calibrator
        """
        pass


class LabelWiseBinaryPredictorMixin(ABC):
    """
    Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
    irrelevant by discretizing the regression scores or probability estimates that are predicted for each label
    individually.
    """

    @abstractmethod
    def use_label_wise_binary_predictor(self) -> LabelWiseBinaryPredictorConfig:
        """
        Configures the rule learner to use a predictor that predicts whether individual labels of given query examples
        are relevant or irrelevant by discretizing the regression scores or probability estimates that are predicted for
        each label individually.

        :return: A `LabelWiseBinaryPredictorConfig` that allows further configuration of the predictor
        """
        pass
            
             
class ExampleWiseBinaryPredictorMixin(ABC):
    """
    Allows to configure a rule learner to use a predictor that predicts known label vectors for given query examples by
    comparing the predicted regression scores or probability estimates to the label vectors encountered in the training
    data.
    """
            
    @abstractmethod
    def use_example_wise_binary_predictor(self) -> ExampleWiseBinaryPredictorConfig:
        """
        Configures the rule learner to use a predictor that predicts known label vectors for given query examples by
        comparing the predicted regression scores or probability estimates to the label vectors encountered in the
        training data.

        :return: An `ExampleWiseBinaryPredictorConfig` that allows further configuration of the predictor
        """
        pass    


class GfmBinaryPredictorMixin(ABC):
    """
    Allows to configure a rule learner to use a predictor that predicts whether individual labels of given query
    examples are relevant or irrelevant by discretizing the regression scores or probability estimates that are
    predicted for each label according to the general F-measure maximizer (GFM).
    """

    @abstractmethod
    def use_gfm_binary_predictor(self) -> GfmBinaryPredictorConfig:
        """
        Configures the rule learner to use a predictor that predicts whether individual labels of given query examples
        are relevant or irrelevant by discretizing the regression scores or probability estimates that are predicted for
        each label according to the general F-measure maximizer (GFM).

        :return: A `GfmBinaryPredictorConfig` that allows further configuration of the predictor
        """
        pass
            
             
class AutomaticBinaryPredictorMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide for a predictor for predicting whether individual labels
    are relevant or irrelevant.
    """

    @abstractmethod
    def use_automatic_binary_predictor(self):
        """
        Configures the rule learner to automatically decide for a predictor for predicting whether individual labels are
        relevant or irrelevant.
        """
        pass
            
             
class LabelWiseScorePredictorMixin(ABC):
    """
    Allows to configure a rule learner to use a predictor that predicts label-wise regression scores for given query
    examples by summing up the scores that are provided by individual rules for each label individually.
    """

    @abstractmethod
    def use_label_wise_score_predictor(self):
        """
        Configures the rule learner to use a predictor that predict label-wise regression scores for given query
        examples by summing up the scores that are provided by individual rules for each label individually.
        """
        pass
            
             
class LabelWiseProbabilityPredictorMixin(ABC):
    """
    Allows to configure a rule learner to use a predictor that predicts label-wise probabilities for given query
    examples by transforming the regression scores that are predicted for each label individually into probabilities.
    """

    @abstractmethod
    def use_label_wise_probability_predictor(self) -> LabelWiseProbabilityPredictorConfig:
        """
        Configures the rule learner to use a predictor that predicts label-wise probabilities for given query examples
        by transforming the regression scores that are predicted for each label individually into probabilities.

        :return: A `LabelWiseProbabilityPredictorConfig` that allows further configuration of the predictor
        """
        pass
            
             
class MarginalizedProbabilityPredictorMixin(ABC):
    """
    Allows to configure a rule learner to use predictor for predicting probability estimates by summing up the scores
    that are provided by individual rules of an existing rule-based model and comparing the aggregated score vector to
    the known label vectors according to a certain distance measure.
    """

    @abstractmethod
    def use_marginalized_probability_predictor(self) -> MarginalizedProbabilityPredictorConfig:
        """
        Configures the rule learner to use a predictor for predicting probability estimates by summing up the scores
        that are provided by individual rules of an existing rule-based model and comparing the aggregated score vector
        to the known label vectors according to a certain distance measure. The probability for an individual label
        calculates as the sum of the distances that have been obtained for all label vectors, where the respective label
        is specified to be relevant, divided by the total sum of all distances.

        :return: A `MarginalizedProbabilityPredictorConfig` that allows further configuration of the predictor
        """
        pass
            
             
class AutomaticProbabilityPredictorMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide for a predictor for predicting probability estimates.
    """
    
    @abstractmethod
    def use_automatic_probability_predictor(self):
        """
        Configures the rule learner to automatically decide for a predictor for predicting probability estimates.
        """
        pass
