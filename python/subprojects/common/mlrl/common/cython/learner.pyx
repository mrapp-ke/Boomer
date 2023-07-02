"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater_or_equal

from cython.operator cimport dereference
from libcpp.utility cimport move

from mlrl.common.cython.feature_info cimport FeatureInfo
from mlrl.common.cython.feature_matrix cimport ColumnWiseFeatureMatrix, RowWiseFeatureMatrix
from mlrl.common.cython.label_matrix cimport RowWiseLabelMatrix
from mlrl.common.cython.label_space_info cimport create_label_space_info
from mlrl.common.cython.prediction cimport BinaryPredictor, ProbabilityPredictor, ScorePredictor, SparseBinaryPredictor
from mlrl.common.cython.probability_calibration cimport create_joint_probability_calibration_model, \
    create_marginal_probability_calibration_model
from mlrl.common.cython.rule_model cimport create_rule_model

from abc import ABC, abstractmethod

from mlrl.common.cython.feature_binning import EqualFrequencyFeatureBinningConfig, EqualWidthFeatureBinningConfig
from mlrl.common.cython.feature_sampling import FeatureSamplingWithoutReplacementConfig
from mlrl.common.cython.instance_sampling import ExampleWiseStratifiedInstanceSamplingConfig, \
    InstanceSamplingWithoutReplacementConfig, InstanceSamplingWithReplacementConfig, \
    LabelWiseStratifiedInstanceSamplingConfig
from mlrl.common.cython.label_sampling import LabelSamplingWithoutReplacementConfig
from mlrl.common.cython.multi_threading import ManualMultiThreadingConfig
from mlrl.common.cython.partition_sampling import ExampleWiseStratifiedBiPartitionSamplingConfig, \
    LabelWiseStratifiedBiPartitionSamplingConfig, RandomBiPartitionSamplingConfig
from mlrl.common.cython.post_optimization import SequentialPostOptimizationConfig
from mlrl.common.cython.rule_induction import BeamSearchTopDownRuleInductionConfig, GreedyTopDownRuleInductionConfig
from mlrl.common.cython.stopping_criterion import PostPruningConfig, PrePruningConfig, SizeStoppingCriterionConfig, \
    TimeStoppingCriterionConfig


cdef class TrainingResult:
    """
    Provides access to the results of fitting a rule learner to training data. It incorporates the model that has been
    trained, as well as additional information that is necessary for obtaining predictions for unseen data.
    """

    def __cinit__(self, uint32 num_labels, RuleModel rule_model not None, LabelSpaceInfo label_space_info not None,
                  MarginalProbabilityCalibrationModel marginal_probability_calibration_model not None,
                  JointProbabilityCalibrationModel joint_probability_calibration_model not None):
        """
        :param num_labels:                              The number of labels for which a model has been trained
        :param rule_model:                              The `RuleModel` that has been trained
        :param label_space_info:                        The `LabelSpaceInfo` that may be used as a basis for making
                                                        predictions
        :param marginal_probability_calibration_model:  The `MarginalProbabilityCalibrationModel` that may be used for
                                                        the calibration of marginal probabilities
        :param joint_probability_calibration_model:     The `JointProbabilityCalibrationModel` that may be used for the
                                                        calibration of joint probabilities    
        """
        self.num_labels = num_labels
        self.rule_model = rule_model
        self.label_space_info = label_space_info
        self.marginal_probability_calibration_model = marginal_probability_calibration_model
        self.joint_probability_calibration_model = joint_probability_calibration_model


cdef class RuleLearner:
    """
    A rule learner.
    """

    cdef IRuleLearner* get_rule_learner_ptr(self):
        pass

    def fit(self, FeatureInfo feature_info not None, ColumnWiseFeatureMatrix feature_matrix not None,
            RowWiseLabelMatrix label_matrix not None, uint32 random_state) -> TrainingResult:
        """
        Applies the rule learner to given training examples and corresponding ground truth labels.

        :param feature_info:    A `FeatureInfo` that provides information about the types of individual features
        :param feature_matrix:  A `ColumnWiseFeatureMatrix` that provides column-wise access to the feature values of
                                the training examples
        :param label_matrix:    A `RowWiseLabelMatrix` that provides row-wise access to the ground truth labels of the
                                training examples
        :param random_state:    The seed to be used by random number generators
        :return:                The `TrainingResult` that provides access to the result of fitting the rule learner to
                                the training data
        """
        assert_greater_or_equal("random_state", random_state, 1)
        cdef unique_ptr[ITrainingResult] training_result_ptr = self.get_rule_learner_ptr().fit(
            dereference(feature_info.get_feature_info_ptr()),
            dereference(feature_matrix.get_column_wise_feature_matrix_ptr()),
            dereference(label_matrix.get_row_wise_label_matrix_ptr()), random_state)
        cdef uint32 num_labels = training_result_ptr.get().getNumLabels()
        cdef unique_ptr[IRuleModel] rule_model_ptr = move(training_result_ptr.get().getRuleModel())
        cdef unique_ptr[ILabelSpaceInfo] label_space_info_ptr = move(training_result_ptr.get().getLabelSpaceInfo())
        cdef unique_ptr[IMarginalProbabilityCalibrationModel] marginal_probability_calibration_model_ptr = \
            move(training_result_ptr.get().getMarginalProbabilityCalibrationModel())
        cdef unique_ptr[IJointProbabilityCalibrationModel] joint_probability_calibration_model_ptr = \
            move(training_result_ptr.get().getJointProbabilityCalibrationModel())
        cdef RuleModel rule_model = create_rule_model(move(rule_model_ptr))
        cdef LabelSpaceInfo label_space_info = create_label_space_info(move(label_space_info_ptr))
        cdef MarginalProbabilityCalibrationModel marginal_probability_calibration_model = \
            create_marginal_probability_calibration_model(move(marginal_probability_calibration_model_ptr))
        cdef JointProbabilityCalibrationModel joint_probability_calibration_model = \
            create_joint_probability_calibration_model(move(joint_probability_calibration_model_ptr))
        return TrainingResult.__new__(TrainingResult, num_labels, rule_model, label_space_info,
                                      marginal_probability_calibration_model, joint_probability_calibration_model)

    def can_predict_binary(self, RowWiseFeatureMatrix feature_matrix not None, uint32 num_labels) -> bool:
        """
        Returns whether the rule learner is able to predict binary labels or not.

        :param feature_matrix:  A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                query examples
        :param num_labels:      The number of labels to predict for
        :return:                True, if the rule learner is able to predict binary labels, False otherwise
        """
        return self.get_rule_learner_ptr().canPredictBinary(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()), num_labels)

    def create_binary_predictor(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                                LabelSpaceInfo label_space_info not None,
                                MarginalProbabilityCalibrationModel marginal_probability_calibration_model not None,
                                JointProbabilityCalibrationModel joint_probability_calibration_model not None,
                                uint32 num_labels) -> BinaryPredictor:
        """
        Creates and returns a predictor that may be used to predict binary labels for given query examples. If the
        prediction of binary labels is not supported by the rule learner, a `RuntimeError` is thrown.

        :param feature_matrix:                          A `RowWiseFeatureMatrix` that provides row-wise access to the
                                                        feature values of the query examples
        :param rule_model:                              The `RuleModel` that should be used to obtain predictions
        :param label_space_info:                        The `LabelSpaceInfo` that provides information about the label
                                                        space that may be used as a basis for obtaining predictions
        :param marginal_probability_calibration_model:  The `MarginalProbabilityCalibrationModel` that may be used for
                                                        the calibration of marginal probabilities
        :param joint_probability_calibration_model:     The `JointProbabilityCalibrationModel` that may be used for the
                                                        calibration of joint probabilities    
        :param num_labels:                              The number of labels to predict for
        :return:                                        A `BinaryPredictor` that may be used to predict binary labels
                                                        for the given query examples
        """
        cdef unique_ptr[IBinaryPredictor] predictor_ptr = move(self.get_rule_learner_ptr().createBinaryPredictor(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
            dereference(rule_model.get_rule_model_ptr()),
            dereference(label_space_info.get_label_space_info_ptr()),
            dereference(marginal_probability_calibration_model.get_marginal_probability_calibration_model_ptr()),
            dereference(joint_probability_calibration_model.get_joint_probability_calibration_model_ptr()),
            num_labels))
        cdef BinaryPredictor binary_predictor = BinaryPredictor.__new__(BinaryPredictor)
        binary_predictor.predictor_ptr = move(predictor_ptr)
        return binary_predictor

    def create_sparse_binary_predictor(self, RowWiseFeatureMatrix feature_matrix not None,
                                       RuleModel rule_model not None, LabelSpaceInfo label_space_info not None,
                                       MarginalProbabilityCalibrationModel marginal_probability_calibration_model not None,
                                       JointProbabilityCalibrationModel joint_probability_calibration_model not None,
                                       uint32 num_labels) -> SparseBinaryPredictor:
        """
        Creates and returns a predictor that may be used to predict sparse binary labels for given query examples. If
        the prediction of sparse binary labels is not supported by the rule learner, a `RuntimeError` is thrown.

        :param feature_matrix:                          A `RowWiseFeatureMatrix` that provides row-wise access to the
                                                        feature values of the query examples
        :param rule_model:                              The `RuleModel` that should be used to obtain predictions
        :param label_space_info:                        The `LabelSpaceInfo` that provides information about the label
                                                        space that may be used as a basis for obtaining predictions
        :param marginal_probability_calibration_model:  The `MarginalProbabilityCalibrationModel` that may be used for
                                                        the calibration of marginal probabilities
        :param joint_probability_calibration_model:     The `JointProbabilityCalibrationModel` that may be used for the
                                                        calibration of joint probabilities                                                            
        :param num_labels:                              The number of labels to predict for
        :return:                                        A `SparseBinaryPredictor` that may be used to predict sparse
                                                        binary labels for the given query examples
        """
        cdef unique_ptr[ISparseBinaryPredictor] predictor_ptr = \
            move(self.get_rule_learner_ptr().createSparseBinaryPredictor(
                dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
                dereference(rule_model.get_rule_model_ptr()),
                dereference(label_space_info.get_label_space_info_ptr()),
                dereference(marginal_probability_calibration_model.get_marginal_probability_calibration_model_ptr()),
                dereference(joint_probability_calibration_model.get_joint_probability_calibration_model_ptr()),
                num_labels))
        cdef SparseBinaryPredictor sparse_binary_predictor = SparseBinaryPredictor.__new__(SparseBinaryPredictor)
        sparse_binary_predictor.predictor_ptr = move(predictor_ptr)
        return sparse_binary_predictor

    def can_predict_scores(self, RowWiseFeatureMatrix feature_matrix not None, uint32 num_labels) -> bool:
        """
        Returns whether the rule learner is able to predict regression scores or not.

        :param feature_matrix:  A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                query examples
        :param num_labels:      The number of labels to predict for
        return:                 True, if the rule learner is able to predict regression scores, False otherwise
        """
        return self.get_rule_learner_ptr().canPredictScores(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()), num_labels)

    def create_score_predictor(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                               LabelSpaceInfo label_space_info not None, uint32 num_labels) -> ScorePredictor:
        """
        Creates and returns a predictor that may be used to predict regression scores for given query examples. If the
        prediction of regression scores is not supported by the rule learner, a `RuntimeError` is thrown.

        :param feature_matrix:      A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                    query examples
        :param rule_model:          The `RuleModel` that should be used to obtain predictions
        :param label_space_info:    The `LabelSpaceInfo` that provides information about the label space that may be
                                    used as a basis for obtaining predictions
        :param num_labels:          The number of labels to predict for
        :return:                    A `ScorePredictor` that may be used to predict regression scores for the given query
                                    examples
        """
        cdef unique_ptr[IScorePredictor] predictor_ptr = move(self.get_rule_learner_ptr().createScorePredictor(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
            dereference(rule_model.get_rule_model_ptr()),
            dereference(label_space_info.get_label_space_info_ptr()),
            num_labels))
        cdef ScorePredictor score_predictor = ScorePredictor.__new__(ScorePredictor)
        score_predictor.predictor_ptr = move(predictor_ptr)
        return score_predictor

    def can_predict_probabilities(self, RowWiseFeatureMatrix feature_matrix not None, uint32 num_labels) -> bool:
        """
        Returns whether the rule learner is able to predict probability estimates or not.

        :param feature_matrix:  A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                query examples
        :param num_labels:      The number of labels to predict for
        :return:                True, if the rule learner is able to predict probability estimates, False otherwise
        """
        return self.get_rule_learner_ptr().canPredictProbabilities(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()), num_labels)

    def create_probability_predictor(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                                     LabelSpaceInfo label_space_info not None,
                                     MarginalProbabilityCalibrationModel marginal_probability_calibration_model not None,
                                     JointProbabilityCalibrationModel joint_probability_calibration_model not None,
                                     uint32 num_labels) -> ProbabilityPredictor:
        """
        Creates and returns a predictor that may be used to predict probability estimates for given query examples. If
        the prediction of probability estimates is not supported by the rule learner, a `RuntimeError` is thrown.

        :param feature_matrix:                          A `RowWiseFeatureMatrix` that provides row-wise access to the
                                                        feature values of the query examples
        :param rule_model:                              The `RuleModel` that should be used to obtain predictions
        :param label_space_info:                        The `LabelSpaceInfo` that provides information about the label
                                                        space that may be used as a basis for obtaining predictions
        :param marginal_probability_calibration_model:  The `MarginalProbabilityCalibrationModel` that may be used for
                                                        the calibration of marginal probabilities
        :param joint_probability_calibration_model:     The `JointProbabilityCalibrationModel` that may be used for the
                                                        calibration of joint probabilities    
        :param num_labels:                              The number of labels to predict for
        :return:                                        A `ProbabilityPredictor` that may be used to predict probability
                                                        estimates for the given query examples
        """
        cdef unique_ptr[IProbabilityPredictor] predictor_ptr = \
            move(self.get_rule_learner_ptr().createProbabilityPredictor(
                dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
                dereference(rule_model.get_rule_model_ptr()),
                dereference(label_space_info.get_label_space_info_ptr()),
                dereference(marginal_probability_calibration_model.get_marginal_probability_calibration_model_ptr()),
                dereference(joint_probability_calibration_model.get_joint_probability_calibration_model_ptr()),
                num_labels))
        cdef ProbabilityPredictor probability_predictor = ProbabilityPredictor.__new__(ProbabilityPredictor)
        probability_predictor.predictor_ptr = move(predictor_ptr)
        return probability_predictor


cdef class RuleLearnerConfig:
    pass


class SequentialRuleModelAssemblageMixin(ABC):
    """
    Allows to configure a rule learner to use an algorithm that sequentially induces several rules.
    """
    
    @abstractmethod
    def use_sequential_rule_model_assemblage(self):
        """
        Configures the rule learner to use an algorithm that sequentially induces several rules, optionally starting
        with a default rule, that are added to a rule-based model.
        """
        pass


class DefaultRuleMixin(ABC):
    """
    Allows to configure a rule learner to induce a default rule.
    """

    @abstractmethod
    def use_default_rule(self):
        """
        Configures the rule learner to induce a default rule.
        """
        pass


class GreedyTopDownRuleInductionMixin(ABC):
    """
    Allows to configure a rule learner to use a greedy top-down search for the induction of individual rules.
    """

    @abstractmethod
    def use_greedy_top_down_rule_induction(self) -> GreedyTopDownRuleInductionConfig:
        """
        Configures the algorithm to use a greedy top-down search for the induction of individual rules.

        :return: A `GreedyTopDownRuleInductionConfig` that allows further configuration of the algorithm for the
                 induction of individual rules
        """
        pass


class BeamSearchTopDownRuleInductionMixin(ABC):
    """
    Allows to configure a rule learner to use a top-down beam search.
    """

    @abstractmethod
    def use_beam_search_top_down_rule_induction(self) -> BeamSearchTopDownRuleInductionConfig:
        """
        Configures the algorithm to use a top-down beam search for the induction of individual rules.

        :return: A `BeamSearchTopDownRuleInductionConfig` that allows further configuration of the algorithm for the
                 induction of individual rules
        """
        pass


class NoPostProcessorMixin(ABC):
    """
    Allows to configure a rule learner to not use any post processor.
    """

    @abstractmethod
    def use_no_post_processor(self):
        """
        Configures the rule learner to not use any post-processor.
        """
        pass


class NoFeatureBinningMixin(ABC):
    """
    Allows to configure a rule learner to not use any method for the assignment of numerical features values to bins.
    """

    @abstractmethod
    def use_no_feature_binning(self):
        """
        Configures the rule learner to not use any method for the assignment of numerical feature values to bins.
        """
        pass


class EqualWidthFeatureBinningMixin(ABC):
    """
    Allows to configure a rule learner to use equal-width feature binning.
    """

    @abstractmethod
    def use_equal_width_feature_binning(self) -> EqualWidthFeatureBinningConfig:
        """
        Configures the rule learner to use a method for the assignment of numerical feature values to bins, such that
        each bin contains values from equally sized value ranges.

        :return: An `EqualWidthFeatureBinningConfig` that allows further configuration of the method for the assignment
                 of numerical feature values to bins
        """
        pass


class EqualFrequencyFeatureBinningMixin(ABC):
    """
    Allows to configure a rule learner to use equal-frequency feature binning.
    """

    @abstractmethod
    def use_equal_frequency_feature_binning(self) -> EqualFrequencyFeatureBinningConfig:
        """
        Configures the rule learner to use a method for the assignment of numerical feature values to bins, such that
        each bin contains approximately the same number of values.

        :return: An `EqualFrequencyFeatureBinningConfig` that allows further configuration of the method for the
                 assignment of numerical feature values to bins
        """
        pass


class NoLabelSamplingMixin(ABC):
    """
    Allows to configure a rule learner to not use label sampling.
    """

    @abstractmethod
    def use_no_label_sampling(self):
        """
        Configures the rule learner to not sample from the available labels whenever a new rule should be learned.
        """
        pass


class RoundRobinLabelSamplingMixin(ABC):
    """
    Allows to configure a rule learner to sample single labels in a round-robin fashion.
    """

    @abstractmethod
    def use_round_robin_label_sampling(self):
        """
        Configures the rule learner to sample a single label in a round-robin fashion whenever a new rule should be
        learned.
        """
        pass


class LabelSamplingWithoutReplacementMixin(ABC):
    """
    Allows to configure a rule learner to use label sampling without replacement.
    """

    @abstractmethod
    def use_label_sampling_without_replacement(self) -> LabelSamplingWithoutReplacementConfig:
        """
        Configures the rule learner to sample from the available labels with replacement whenever a new rule should be
        learned.

        :return: A `LabelSamplingWithoutReplacementConfig` that allows further configuration of the method for sampling
                 labels
        """
        pass


class NoInstanceSamplingMixin(ABC):
    """
    Defines an interface for all classes that allow to configure a rule learner to not use instance sampling.
    """

    @abstractmethod
    def use_no_instance_sampling(self):
        """
        Configures the rule learner to not sample from the available training examples whenever a new rule should be
        learned.
        """
        pass


class InstanceSamplingWithReplacementMixin(ABC):
    """
    Defines an interface for all classes that allow to configure a rule learner to use instance sampling with
    replacement.
    """

    @abstractmethod
    def use_instance_sampling_with_replacement(self) -> InstanceSamplingWithReplacementConfig:
        """
        Configures the rule learner to sample from the available training examples with replacement whenever a new rule
        should be learned.

        :return: An `InstanceSamplingWithReplacementConfig` that allows further configuration of the method for sampling
                 instances
        """
        pass


class InstanceSamplingWithoutReplacementMixin(ABC):
    """
    Defines an interface for all classes that allow to configure a rule learner to use instance sampling without
    replacement.
    """

    @abstractmethod
    def use_instance_sampling_without_replacement(self) -> InstanceSamplingWithoutReplacementConfig:
        """
        Configures the rule learner to sample from the available training examples without replacement whenever a new
        rule should be learned.

        :return: An `InstanceSamplingWithoutReplacementConfig` that allows further configuration of the method for
                 sampling instances
        """
        pass


class LabelWiseStratifiedInstanceSamplingMixin(ABC):
    """
    Allows to configure a rule learner to use label-wise stratified instance sampling.
    """

    @abstractmethod
    def use_label_wise_stratified_instance_sampling(self) -> LabelWiseStratifiedInstanceSamplingConfig:
        """
        Configures the rule learner to sample from the available training examples using stratification, such that for
        each label the proportion of relevant and irrelevant examples is maintained, whenever a new rule should be
        learned.

        :return: A `LabelWiseStratifiedInstanceSamplingConfig` that allows further configuration of the method for
                 sampling instances
        """
        pass


class ExampleWiseStratifiedInstanceSamplingMixin(ABC):
    """
    Allows to configure a rule learner to use example-wise stratified instance sampling.
    """

    @abstractmethod
    def use_example_wise_stratified_instance_sampling(self) -> ExampleWiseStratifiedInstanceSamplingConfig:
        """
        Configures the rule learner to sample from the available training examples using stratification, where distinct
        label vectors are treated as individual classes, whenever a new rule should be learned.

        :return: An `ExampleWiseStratifiedInstanceSamplingConfig` that allows further configuration of the method for
                 sampling instances
        """
        pass


class NoFeatureSamplingMixin(ABC):
    """
    Allows to configure a rule learner to not use feature sampling.
    """

    @abstractmethod
    def use_no_feature_sampling(self):
        """
        Configures the rule learner to not sample from the available features whenever a rule should be refined.
        """
        pass
        

class FeatureSamplingWithoutReplacementMixin(ABC):
    """
    Allows to configure a rule learner to use feature sampling without replacement.
    """

    @abstractmethod
    def use_feature_sampling_without_replacement(self) -> FeatureSamplingWithoutReplacementConfig:
        """
        Configures the rule learner to sample from the available features with replacement whenever a rule should be
        refined.

        :return: A `FeatureSamplingWithoutReplacementConfig` that allows further configuration of the method for
                 sampling features
        """
        pass


class NoPartitionSamplingMixin(ABC):
    """
    Allows to configure a rule learner to not partition the available training examples into a training set and a
    holdout set.
    """

    @abstractmethod
    def use_no_partition_sampling(self):
        """
        Configures the rule learner to not partition the available training examples into a training set and a holdout
        set.
        """
        pass


class RandomBiPartitionSamplingMixin(ABC):
    """
    Allows to configure a rule learner to partition the available training example into a training set and a holdout set
    by randomly splitting the training examples into two mutually exclusive sets.
    """

    @abstractmethod
    def use_random_bi_partition_sampling(self) -> RandomBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        by randomly splitting the training examples into two mutually exclusive sets.

        :return: A `RandomBiPartitionSamplingConfig` that allows further configuration of the method for partitioning
                 the available training examples into a training set and a holdout set
        """
        pass


class LabelWiseStratifiedBiPartitionSamplingMixin(ABC):
    """
    Allows to configure a rule learner to partition the available training examples into a training set and a holdout
    set using stratification, such that for each label the proportion of relevant and irrelevant examples is maintained.
    """

    @abstractmethod
    def use_label_wise_stratified_bi_partition_sampling(self) -> LabelWiseStratifiedBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        using stratification, such that for each label the proportion of relevant and irrelevant examples is maintained.

        :return: A `LabelWiseStratifiedBiPartitionSamplingConfig` that allows further configuration of the method for
                 partitioning the available training examples into a training and a holdout set
        """
        pass


class ExampleWiseStratifiedBiPartitionSamplingMixin(ABC):
    """
    Allows to configure a rule learner to partition the available training examples into a training set and a holdout
    set using stratification, where distinct label vectors are treated as individual classes.
    """

    @abstractmethod
    def use_example_wise_stratified_bi_partition_sampling(self) -> ExampleWiseStratifiedBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        using stratification, where distinct label vectors are treated as individual classes

        :return: An `ExampleWiseStratifiedBiPartitionSamplingConfig` that allows further configuration of the method for
                 partitioning the available training examples into a training and a holdout set
        """
        pass


class NoRulePruningMixin(ABC):
    """
    Allows to configure a rule learner to not prune individual rules.
    """

    @abstractmethod
    def use_no_rule_pruning(self):
        """
        Configures the rule learner to not prune individual rules.
        """
        pass


class IrepRulePruningMixin(ABC):
    """
    Allows to configure a rule learner to prune individual rules by following the principles of "incremental reduced
    error pruning" (IREP).
    """

    @abstractmethod
    def use_irep_rule_pruning(self):
        """
        Configures the rule learner to prune individual rules by following the principles of "incremental reduced error
        pruning" (IREP).
        """
        pass


class NoParallelRuleRefinementMixin(ABC):
    """
    Allows to configure a rule learner to not use any multi-threading for the parallel refinement of rules.
    """

    @abstractmethod
    def use_no_parallel_rule_refinement(self):
        """
        Configures the rule learner to not use any multi-threading for the parallel refinement of rules.
        """
        pass


class ParallelRuleRefinementMixin(ABC):
    """
    Allows to configure a rule learner to use multi-threading for the parallel refinement of rules.
    """

    @abstractmethod
    def use_parallel_rule_refinement(self) -> ManualMultiThreadingConfig:
        """
        Configures the rule learner to use multi-threading for the parallel refinement of rules.

        :return: A `ManualMultiThreadingConfig` that allows further configuration of the multi-threading behavior
        """
        pass


class NoParallelStatisticUpdateMixin(ABC):
    """
    Allows to configure a rule learner to not use any multi-threading for the parallel update of statistics.
    """

    @abstractmethod
    def use_no_parallel_statistic_update(self):
        """
        Configures the rule learner to not use any multi-threading for the parallel update of statistics.
        """
        pass


class ParallelStatisticUpdateMixin(ABC):
    """
    Allows to configure a rule learner to use multi-threading for the parallel update of statistics.
    """

    @abstractmethod
    def use_parallel_statistic_update(self) -> ManualMultiThreadingConfig:
        """
        Configures the rule learner to use multi-threading for the parallel update of statistics.

        :return: A `ManualMultiThreadingConfig` that allows further configuration of the multi-threading behavior
        """
        pass


class NoParallelPredictionMixin(ABC):
    """
    Allows to configure a rule learner to not use any multi-threading for prediction.
    """

    @abstractmethod
    def use_no_parallel_prediction(self):
        """
        Configures the rule learner to not use any multi-threading to predict for several query examples in parallel.
        """
        pass


class ParallelPredictionMixin(ABC):
    """
    Allows to configure a rule learner to use multi-threading to predict for several examples in parallel.
    """

    @abstractmethod
    def use_parallel_prediction(self) -> ManualMultiThreadingConfig:
        """
        Configures the rule learner to use multi-threading to predict for several query examples in parallel.

        :return: A `ManualMultiThreadingConfig` that allows further configuration of the multi-threading behavior
        """
        pass


class NoSizeStoppingCriterionMixin(ABC):
    """
    Allows to configure a rule learner to not use a stopping criterion that ensures that the number of induced rules
    does not exceed a certain maximum.
    """

    @abstractmethod
    def use_no_size_stopping_criterion(self):
        """
        Configures the rule learner to not use a stopping criterion that ensures that the number of induced rules does
        not exceed a certain maximum.
        """
        pass


class SizeStoppingCriterionMixin(ABC):
    """
    Allows to configure a rule learner to use a stopping criterion that ensures that the number of induced rules does
    not exceed a certain maximum.
    """

    @abstractmethod
    def use_size_stopping_criterion(self) -> SizeStoppingCriterionConfig:
        """
        Configures the rule learner to use a stopping criterion that ensures that the number of induced rules does not
        exceed a certain maximum.

        :return: A `SizeStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        pass


class NoTimeStoppingCriterionMixin(ABC):
    """
    Allows to configure a rule learner to not use a stopping criterion that ensures that a certain time limit is not
    exceeded.
    """

    @abstractmethod
    def use_no_time_stopping_criterion(self):
        """
        Configures the rule learner to not use a stopping criterion that ensures that a certain time limit is not
        exceeded.
        """
        pass


class TimeStoppingCriterionMixin(ABC):
    """
    Allows to configure a rule learner to use a stopping criterion that ensures that a certain time limit is not
    exceeded.
    """

    def use_time_stopping_criterion(self) -> TimeStoppingCriterionConfig:
        """
        Configures the rule learner to use a stopping criterion that ensures that a certain time limit is not exceeded.

        :return: A `TimeStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        pass


class PrePruningMixin(ABC):
    """
    Allows to configure a rule learner to use a stopping criterion that stops the induction of rules as soon as the
    quality of a model's predictions for the examples in the training or holdout set do not improve according to a
    certain measure.
    """

    @abstractmethod
    def use_global_pre_pruning(self) -> PrePruningConfig:
        """
        Configures the rule learner to use a stopping criterion that stops the induction of rules as soon as the quality
        of a model's predictions for the examples in the training or holdout set do not improve according to a certain
        measure.

        :return: A `PrePruningConfig` that allows further configuration of the stopping criterion
        """
        pass


class NoGlobalPruningMixin(ABC):
    """
    Allows to configure a rule learner to not use global pruning.
    """

    @abstractmethod
    def use_no_global_pruning(self):
        """
        Configures the rule learner to not use global pruning.
        """
        pass


class PostPruningMixin(ABC):
    """
    Allows to configure a rule learner to use a stopping criterion that keeps track of the number of rules in a model
    that perform best with respect to the examples in the training or holdout set according to a certain measure.
    """

    @abstractmethod
    def use_global_post_pruning(self) -> PostPruningConfig:
        """
        Configures the rule learner to use a stopping criterion that keeps track of the number of rules in a model that
        perform best with respect to the examples in the training or holdout set according to a certain measure.
        """
        pass


class NoSequentialPostOptimizationMixin(ABC):
    """
    Allows to configure a rule learner to not use a post-optimization method that optimizes each rule in a model by
    relearning it in the context of the other rules.
    """

    @abstractmethod
    def use_no_sequential_post_optimization(self):
        """
        Configures the rule learner to not use a post-optimization method that optimizes each rule in a model by
        relearning it in the context of the other rules.
        """
        pass


class SequentialPostOptimizationMixin(ABC):
    """
    Allows to configure a rule learner to use a post-optimization method that optimizes each rule in a model by
    relearning it in the context of the other rules.
    """

    @abstractmethod
    def use_sequential_post_optimization(self) -> SequentialPostOptimizationConfig:
        """
        Configures the rule learner to use a post-optimization method that optimizes each rule in a model by relearning
        it in the context of the other rules.

        :return: A `SequentialPostOptimizationConfig` that allows further configuration of the post-optimization method
        """
        pass


class NoMarginalProbabilityCalibrationMixin(ABC):
    """
    Allows to configure a rule learner to not calibrate marginal probabilities.
    """

    @abstractmethod
    def use_no_marginal_probability_calibration(self):
        """
        Configures the rule learner to not calibrate marginal probabilities.
        """
        pass


class NoJointProbabilityCalibrationMixin(ABC):
    """
    Allows to configure a rule learner to not calibrate joint probabilities.
    """
     
    @abstractmethod
    def use_no_joint_probability_calibration(self):
        """
        Configures the rule learner to not calibrate joint probabilities.
        """
        pass
