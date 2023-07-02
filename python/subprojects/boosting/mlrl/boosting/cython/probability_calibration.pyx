"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class IsotonicMarginalProbabilityCalibratorConfig:
    """
    Allows to configure a calibrator that fits a model for the calibration of marginal probabilities via isotonic
    regression.
    """

    def is_holdout_set_used(self) -> bool:
        """
        Returns whether the calibration model is fit to the examples in the holdout set, if available, or not.

        :return: True, if the calibration model is fit to the examples in the holdout set, if available, False if the
                 training set is used instead
        """
        return self.config_ptr.isHoldoutSetUsed()

    def set_use_holdout_set(self, use_holdout_set: bool) -> IsotonicMarginalProbabilityCalibratorConfig:
        """
        Sets whether the calibration model should be fit to the examples in the holdout set, if available, or not.

        :param use_holdout_set: True, if the calibration model should be fit to the examples in the holdout set, if
                                available, False if the training set should be used instead
        :return:                An `IsotonicMarginalProbabilityCalibratorConfig` that allows further configuration of
                                the calibrator
        """
        self.config_ptr.setUseHoldoutSet(use_holdout_set)
        return self


cdef class IsotonicJointProbabilityCalibratorConfig:
    """
    Allows to configure a calibrator that fits a model for the calibration of joint probabilities via isotonic
    regression.
    """

    def is_holdout_set_used(self) -> bool:
        """
        Returns whether the calibration model is fit to the examples in the holdout set, if available, or not.

        :return: True, if the calibration model is fit to the examples in the holdout set, if available, False if the
                 training set is used instead
        """
        return self.config_ptr.isHoldoutSetUsed()

    def set_use_holdout_set(self, use_holdout_set: bool) -> IsotonicJointProbabilityCalibratorConfig:
        """
        Sets whether the calibration model should be fit to the examples in the holdout set, if available, or not.

        :param use_holdout_set: True, if the calibration model should be fit to the examples in the holdout set, if
                                available, False if the training set should be used instead
        :return:                An `IsotonicJointProbabilityCalibratorConfig` that allows further configuration of the
                                calibrator
        """
        self.config_ptr.setUseHoldoutSet(use_holdout_set)
        return self
