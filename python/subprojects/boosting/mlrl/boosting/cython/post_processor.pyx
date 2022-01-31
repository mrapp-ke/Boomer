"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._validation import assert_greater, assert_less


cdef class ConstantShrinkageConfig:
    """
    Allows to configure a post-processor that shrinks the weights of rules by a constant "shrinkage" parameter.
    """

    def get_shrinkage(self) -> float:
        """
        Returns the value of the "shrinkage" parameter.

        :return: The value of the "shrinkage" parameter
        """
        return self.config_ptr.getShrinkage()

    def set_shrinkage(self, shrinkage: float) -> ConstantShrinkageConfig:
        """
        Sets the value of the "shrinkage" parameter.

        :param shrinkage:   The value of the "shrinkage" parameter. Must be in (0, 1)
        :return:            A `ConstantShrinkageConfig` that allows further configuration of the post-processor
        """
        assert_greater('shrinkage', shrinkage, 0)
        assert_less('shrinkage', shrinkage, 1)
        self.config_ptr.setShrinkage(shrinkage)
        return self
