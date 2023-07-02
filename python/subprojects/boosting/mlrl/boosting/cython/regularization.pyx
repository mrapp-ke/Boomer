"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater


cdef class ManualRegularizationConfig:
    """
    Allows to configure a regularization term that affects the evaluation of rules by manually specifying the weight of
    the regularization term.
    """

    def get_regularization_weight(self) -> float:
        """
        Returns the weight of the regularization term.

        :return: The weight of the regularization term
        """
        return self.config_ptr.getRegularizationWeight()

    def set_regularization_weight(self, regularization_weight: float) -> ManualRegularizationConfig:
        """
        Sets the weight of the regularization term.

        :param regularization_weight:   The weight of the regularization term. Must be greater than 0
        :return:                        A `ManualRegularizationConfig` that allows further configuration of the
                                        regularization term
        """
        assert_greater('regularization_weight', regularization_weight, 0)
        self.config_ptr.setRegularizationWeight(regularization_weight)
        return self
