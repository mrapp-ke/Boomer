#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides a scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
classification rules. The classifier is composed of several modules, e.g., for rule induction and prediction.
"""
from abc import abstractmethod

import numpy as np
from boomer.algorithm._head_refinement import HeadRefinement, SingleLabelHeadRefinement, FullHeadRefinement
from boomer.algorithm._losses import Loss, DecomposableLoss
from boomer.algorithm._pruning import Pruning, IREP

from boomer.algorithm._example_based_losses import ExampleBasedLogisticLoss
from boomer.algorithm._macro_losses import MacroSquaredErrorLoss, MacroLogisticLoss
from boomer.algorithm._shrinkage import Shrinkage, ConstantShrinkage
from boomer.algorithm._sub_sampling import FeatureSubSampling, RandomFeatureSubsetSelection
from boomer.algorithm._sub_sampling import InstanceSubSampling, Bagging, RandomInstanceSubsetSelection
from boomer.algorithm._sub_sampling import LabelSubSampling, RandomLabelSubsetSelection
from boomer.algorithm.model import DTYPE_FLOAT32
from boomer.algorithm.prediction import Prediction, Sign, LinearCombination
from boomer.algorithm.rule_induction import RuleInduction, GradientBoosting
from boomer.algorithm.stopping_criteria import SizeStoppingCriterion, TimeStoppingCriterion
from boomer.learners import MLLearner
from boomer.stats import Stats


class MLRuleLearner(MLLearner):
    """
    A scikit-multilearn implementation of a rule learner algorithm for multi-label classification or ranking.

    Attributes
        stats_          Statistics about the training data set
        theory_         The theory that contains the classification rules
        persistence     The 'ModelPersistence' to be used to load/save the theory
    """

    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        self.require_dense = [True, True]  # We need a dense representation of the training data

    def get_model_prefix(self) -> str:
        return 'rules'

    def _fit(self, stats: Stats, x: np.ndarray, y: np.ndarray, random_state: int):
        # Create a dense representation of the training data
        x = self._ensure_input_format(x)
        y = self._ensure_input_format(y)

        # Induce rules
        rule_induction = self._create_rule_induction(stats)
        rule_induction.random_state = random_state
        theory = rule_induction.induce_rules(stats, x, y)
        return theory

    def _predict(self, model, stats: Stats, x: np.ndarray, random_state: int) -> np.ndarray:
        # Create a dense representation of the given examples
        x = self._ensure_input_format(x)

        # Convert feature matrix into Fortran-contiguous array
        x = np.asfortranarray(x, dtype=DTYPE_FLOAT32)

        prediction = self._create_prediction()
        prediction.random_state = self.random_state
        return prediction.predict(stats, model, x)

    @abstractmethod
    def _create_prediction(self) -> Prediction:
        """
        Must be implemented by subclasses in order to create the `Prediction` to be used for making predictions.

        :return: The `Prediction` that has been created
        """
        pass

    @abstractmethod
    def _create_rule_induction(self, stats: Stats) -> RuleInduction:
        """
        Must be implemented by subclasses in order to create the `RuleInduction` to be used for inducing rules.

        :param stats:   Statistics about the training data set
        :return:        The `RuleInduction` that has been created
        """
        pass


class Boomer(MLRuleLearner):
    """
    A scikit-multilearn implementation of "BOOMER" -- an algorithm for learning gradient boosted multi-label
    classification rules.
    """

    def __init__(self, model_dir: str = None, num_rules: int = 500, time_limit: int = -1, head_refinement: str = None,
                 loss: str = 'macro-squared-error-loss', label_sub_sampling: int = -1,
                 instance_sub_sampling: str = None, feature_sub_sampling: str = None, pruning: str = None,
                 shrinkage: float = 1.0, l2_regularization_weight: float = 0.0):
        """
        :param num_rules:                   The number of rules to be induced (including the default rule)
        :param time_limit:                  The duration in seconds after which the induction of rules should be
                                            canceled
        :param head_refinement:             The strategy that is used to find the heads of rules. Must be
                                            `single-label`, `full` or None, if the default strategy should be used
        :param loss:                        The loss function to be minimized. Must be `macro-squared-error-loss` or
                                            `example-based-logistic-loss`
        :param label_sub_sampling:          The number of samples to be used for sub-sampling the labels each time a new
                                            classification rule is learned. Must be at least 1 or -1, if no sub-sampling
                                            should be used
        :param instance_sub_sampling:       The strategy that is used for sub-sampling the training examples each time a
                                            new classification rule is learned. Must be `bagging`,
                                            `random-instance-selection` or None, if no sub-sampling should be used
        :param feature_sub_sampling:        The strategy that is used for sub-sampling the features each time a
                                            classification rule is refined. Must be `random-feature-selection` or None,
                                            if no sub-sampling should be used
        :param pruning:                     The strategy that is used for pruning rules. Must be `irep` or None, if no
                                            pruning should be used
        :param shrinkage:                   The shrinkage parameter that should be applied to the predictions of newly
                                            induced rules to reduce their effect on the entire model. Must be in (0, 1]
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores that are predicted by rules. Must be at least 0
        """
        super().__init__(model_dir)
        self.num_rules = num_rules
        self.time_limit = time_limit
        self.head_refinement = head_refinement
        self.loss = loss
        self.label_sub_sampling = label_sub_sampling
        self.instance_sub_sampling = instance_sub_sampling
        self.feature_sub_sampling = feature_sub_sampling
        self.pruning = pruning
        self.shrinkage = shrinkage
        self.l2_regularization_weight = l2_regularization_weight

    def _create_prediction(self) -> Prediction:
        return Sign(LinearCombination())

    def _create_rule_induction(self, stats: Stats) -> RuleInduction:
        num_rules = int(self.num_rules)
        time_limit = int(self.time_limit)
        stopping_criteria = []

        if num_rules != -1:
            if num_rules > 0:
                stopping_criteria.append(SizeStoppingCriterion(num_rules))
            else:
                raise ValueError('Invalid value given for parameter \'num_rules\': ' + str(num_rules))

        if time_limit != -1:
            if time_limit > 0:
                stopping_criteria.append(TimeStoppingCriterion(time_limit))
            else:
                raise ValueError('Invalid value given for parameter \'time_limit\': ' + str(time_limit))

        l2_regularization_weight = float(self.l2_regularization_weight)

        if l2_regularization_weight < 0:
            raise ValueError(
                'Invalid value given for parameter \'l2_regularization_weight\': ' + str(l2_regularization_weight))

        loss = self.__create_loss(l2_regularization_weight)
        head_refinement = self.__create_head_refinement(loss)
        label_sub_sampling = self.__create_label_sub_sampling(stats)
        instance_sub_sampling = self.__create_instance_sub_sampling()
        feature_sub_sampling = self.__create_feature_sub_sampling()
        pruning = self.__create_pruning()
        shrinkage = self.__create_shrinkage()
        return GradientBoosting(head_refinement, loss, label_sub_sampling, instance_sub_sampling, feature_sub_sampling,
                                pruning, shrinkage, *stopping_criteria)

    def __create_loss(self, l2_regularization_weight: float) -> Loss:
        loss = self.loss

        if loss == 'macro-squared-error-loss':
            return MacroSquaredErrorLoss(l2_regularization_weight)
        elif loss == 'macro-logistic-loss':
            return MacroLogisticLoss(l2_regularization_weight)
        elif loss == 'example-based-logistic-loss':
            return ExampleBasedLogisticLoss(l2_regularization_weight)
        raise ValueError('Invalid value given for parameter \'loss\': ' + str(loss))

    def __create_head_refinement(self, loss: Loss) -> HeadRefinement:
        head_refinement = self.head_refinement

        if head_refinement is None:
            return SingleLabelHeadRefinement() if isinstance(loss, DecomposableLoss) else FullHeadRefinement()
        elif head_refinement == 'single-label':
            return SingleLabelHeadRefinement()
        elif head_refinement == 'full':
            return FullHeadRefinement()
        raise ValueError('Invalid value given for parameter \'head_refinement\': ' + str(head_refinement))

    def __create_label_sub_sampling(self, stats: Stats) -> LabelSubSampling:
        label_sub_sampling = int(self.label_sub_sampling)

        if label_sub_sampling == -1:
            return None
        elif label_sub_sampling > 0:
            if label_sub_sampling < stats.num_labels:
                return RandomLabelSubsetSelection(label_sub_sampling)
            else:
                raise ValueError('Value given for parameter \'label_sub_sampling\' (' + str(label_sub_sampling)
                                 + ') must be less that the number of labels in the training data set ('
                                 + str(stats.num_labels) + ')')
        raise ValueError('Invalid value given for parameter \'label_sub_sampling\': ' + str(label_sub_sampling))

    def __create_instance_sub_sampling(self) -> InstanceSubSampling:
        instance_sub_sampling = self.instance_sub_sampling

        if instance_sub_sampling is None:
            return None
        elif instance_sub_sampling == 'bagging':
            return Bagging()
        elif instance_sub_sampling == 'random-instance-selection':
            return RandomInstanceSubsetSelection()
        raise ValueError('Invalid value given for parameter \'instance_sub_sampling\': ' + str(instance_sub_sampling))

    def __create_feature_sub_sampling(self) -> FeatureSubSampling:
        feature_sub_sampling = self.feature_sub_sampling

        if feature_sub_sampling is None:
            return None
        elif feature_sub_sampling == 'random-feature-selection':
            return RandomFeatureSubsetSelection()
        raise ValueError('Invalid value given for parameter \'feature_sub_sampling\': ' + str(feature_sub_sampling))

    def __create_pruning(self) -> Pruning:
        pruning = self.pruning

        if pruning is None:
            return None
        if pruning == 'irep':
            return IREP()
        raise ValueError('Invalid value given for parameter \'pruning\': ' + str(pruning))

    def __create_shrinkage(self) -> Shrinkage:
        shrinkage = float(self.shrinkage)

        if 0.0 < shrinkage < 1.0:
            return ConstantShrinkage(shrinkage)
        if shrinkage == 1.0:
            return None
        raise ValueError('Invalid value given for parameter \'shrinkage\': ' + str(shrinkage))

    def get_name(self) -> str:
        name = 'num-rules=' + str(self.num_rules)
        if self.head_refinement is not None:
            name += '_head-refinement=' + str(self.head_refinement)
        name += '_loss=' + str(self.loss)
        if self.label_sub_sampling is not None:
            name += '_label-sub-sampling=' + str(self.label_sub_sampling)
        if self.instance_sub_sampling is not None:
            name += '_instance-sub-sampling=' + str(self.instance_sub_sampling)
        if self.feature_sub_sampling is not None:
            name += '_feature-sub-sampling=' + str(self.feature_sub_sampling)
        if self.pruning is not None:
            name += '_pruning=' + str(self.pruning)
        if 0.0 < float(self.shrinkage) < 1.0:
            name += '_shrinkage=' + str(self.shrinkage)
        if float(self.l2_regularization_weight) > 0.0:
            name += '_l2=' + str(self.l2_regularization_weight)
        return name

    def get_params(self, deep=True):
        params = super().get_params()
        params.update({
            'num_rules': self.num_rules,
            'time_limit': self.time_limit,
            'head_refinement': self.head_refinement,
            'loss': self.loss,
            'label_sub_sampling': self.label_sub_sampling,
            'instance_sub_sampling': self.instance_sub_sampling,
            'feature_sub_sampling': self.feature_sub_sampling,
            'pruning': self.pruning,
            'shrinkage': self.shrinkage,
            'l2_regularization_weight': self.l2_regularization_weight
        })
        return params
