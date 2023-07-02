"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater_or_equal


cdef class SequentialPostOptimizationConfig:
    """
    Allows to configure a method that optimizes each rule in a model by relearning it in the context of the other rules.
    Multiple iterations, where the rules in a model are relearned in the order of their induction, may be carried out.
    """

    def get_num_iterations(self) -> int:
        """
        Returns the number of iterations that are performed for optimizing a model.

        :return: The number of iterations that are performed for optimizing a model
        """
        return self.config_ptr.getNumIterations()

    def set_num_iterations(self, num_iterations: int) -> SequentialPostOptimizationConfig:
        """
        Sets the number of iterations that should be performed for optimizing a model.

        :param num_iterations:  The number of iterations to be performed. Must be at least 1
        :return:                An `SequentialPostOptimizationConfig` that allows further configuration of the
                                optimization method
        """
        assert_greater_or_equal('num_iterations', num_iterations, 1)
        self.config_ptr.setNumIterations(num_iterations)
        return self

    def are_heads_refined(self) -> bool:
        """
        Returns whether the heads of rules are refined when being relearned or not.

        :return: True, if the heads of rules are refined when being relearned, False otherwise
        """
        return self.config_ptr.areHeadsRefined()

    def set_refine_heads(self, refine_heads: bool) -> SequentialPostOptimizationConfig:
        """
        Sets whether the heads of rules should be refined when being relearned or not.

        :param refine_heads:    True, if the heads of rules should be refined when being relearned, False otherwise
        :return:                An `SequentialPostOptimizationConfig` that allows further configuration of the
                                optimization method
        """
        self.config_ptr.setRefineHeads(refine_heads)
        return self

    def are_features_resampled(self) -> bool:
        """
        Returns whether a new sample of the available features is created whenever a new rule is refined or not.

        :return: True, if a new sample of the available features is created whenever a new rule is refined, false, if
                 the conditions of the new rule use the same features as the original rule
        """
        return self.config_ptr.areFeaturesResampled()

    def set_resample_features(self, resample_features: bool) -> SequentialPostOptimizationConfig:
        """
        Sets whether a new sample of the available features should be created whenever a new rule is refined or not.

        :param resample_features:   True, if a new sample of the available features should be created whenever a new
                                    rule is refined, false, if the conditions of the new rule should use the same
                                    features as the original rule
        :return:                    An `SequentialPostOptimizationConfig` that allows further configuration of the
                                    optimization method
        """
        self.config_ptr.setResampleFeatures(resample_features)
        return self
