#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for inducing classification rules.
"""
import logging as log
from abc import abstractmethod

import numpy as np
from boomer.algorithm._head_refinement import HeadRefinement
from boomer.algorithm._losses import Loss
from boomer.algorithm._pruning import Pruning

from boomer.algorithm._rule_induction import induce_default_rule, induce_rule
from boomer.algorithm._shrinkage import Shrinkage
from boomer.algorithm._sub_sampling import InstanceSubSampling, FeatureSubSampling, LabelSubSampling
from boomer.algorithm.model import Theory, DTYPE_INTP, DTYPE_UINT8, DTYPE_FLOAT32
from boomer.algorithm.stopping_criteria import StoppingCriterion
from boomer.interfaces import Randomized
from boomer.stats import Stats


class RuleInduction(Randomized):
    """
    A module that allows to induce a `Theory`, consisting of several classification rules.
    """

    @abstractmethod
    def induce_rules(self, stats: Stats, x: np.ndarray, y: np.ndarray) -> Theory:
        """
        Creates and returns a 'Theory' that contains several candidate rules.

        :param stats:   Statistics about the training data set
        :param x:       An array of dtype float, shape `(num_examples, num_features)`, representing the features of the
                        training examples
        :param y:       An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the
                        training examples
        :return:        A 'Theory' that contains the induced classification rules
        """
        pass


class GradientBoosting(RuleInduction):
    """
    Implements the induction of (multi-label) classification rules using gradient boosting.
    """

    def __init__(self, head_refinement: HeadRefinement, loss: Loss, label_sub_sampling: LabelSubSampling,
                 instance_sub_sampling: InstanceSubSampling, feature_sub_sampling: FeatureSubSampling, pruning: Pruning,
                 shrinkage: Shrinkage, *stopping_criteria: StoppingCriterion):
        """
        :param head_refinement:         The strategy that is used to find the heads of rules
        :param loss:                    The loss function to be minimized
        :param label_sub_sampling:      The strategy that is used for sub-sampling the labels each time a new
                                        classification rule is learned
        :param instance_sub_sampling:   The strategy that is used for sub-sampling the training examples each time a new
                                        classification rule is learned
        :param feature_sub_sampling:    The strategy that is used for sub-sampling the features each time a
                                        classification rule is refined
        :param pruning:                 The strategy that is used for pruning rules
        :param shrinkage:               The shrinkage parameter that should be applied to the predictions of newly
                                        induced rules to reduce their effect on the entire model. Must be in (0, 1]
        :param stopping_criteria        The stopping criteria that should be used to decide whether additional rules
                                        should be induced or not
        """
        self.head_refinement = head_refinement
        self.loss = loss
        self.label_sub_sampling = label_sub_sampling
        self.instance_sub_sampling = instance_sub_sampling
        self.feature_sub_sampling = feature_sub_sampling
        self.pruning = pruning
        self.shrinkage = shrinkage
        self.stopping_criteria = stopping_criteria

    def induce_rules(self, stats: Stats, x: np.ndarray, y: np.ndarray) -> Theory:
        self.__validate()
        stopping_criteria = self.stopping_criteria
        random_state = self.random_state
        head_refinement = self.head_refinement
        loss = self.loss
        label_sub_sampling = self.label_sub_sampling
        instance_sub_sampling = self.instance_sub_sampling
        feature_sub_sampling = self.feature_sub_sampling
        pruning = self.pruning
        shrinkage = self.shrinkage

        # Convert feature and label matrices into Fortran-contiguous arrays
        x = np.asfortranarray(x, dtype=DTYPE_FLOAT32)
        y = np.asfortranarray(y, dtype=DTYPE_UINT8)

        # Sort feature matrix once
        x_sorted_indices = np.asfortranarray(np.argsort(x, axis=0), dtype=DTYPE_INTP)

        # Create a new theory
        theory = []

        # Induce default rule, if necessary
        if len(theory) == 0:
            log.info('Learning rule 1(default rule)...')
            default_rule = induce_default_rule(y, loss)
            theory.append(default_rule)

        while all([stopping_criterion.should_continue(theory) for stopping_criterion in stopping_criteria]):
            log.info('Learning rule %s...', len(theory) + 1)

            # Induce a new rule
            rule = induce_rule(x, x_sorted_indices, y, head_refinement, loss, label_sub_sampling, instance_sub_sampling,
                               feature_sub_sampling, pruning, shrinkage, random_state)

            # Add new rule to theory
            theory.append(rule)

            # Alter random state for inducing the next rule
            random_state += 1

        return theory

    def __validate(self):
        """
        Raises exceptions if the module is not configured properly.
        """

        if self.stopping_criteria is None or len(self.stopping_criteria) < 1:
            raise ValueError('Number of \'stopping_criteria\' must be at least 1')
        if self.head_refinement is None:
            raise ValueError('Parameter \'head_refinement\' may not be None')
