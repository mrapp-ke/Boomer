"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class LabelWiseProbabilityPredictorConfig:
    """
    Allows to configure a predictor that predicts label-wise probabilities for given query examples, which estimate the
    chance of individual labels to be relevant, by summing up the scores that are provided by individual rules of an
    existing rule-based model and transforming the aggregated scores into probabilities in [0, 1] according to a certain
    transformation function that is applied to each label individually.
    """

    def is_probability_calibration_model_used(self) -> bool:
        """
        Returns whether a model for the calibration of probabilities is used, if available, or not.

        :return: True, if a model for the calibration of probabilities is used, if available, False otherwise
        """
        return self.config_ptr.isProbabilityCalibrationModelUsed()

    def set_use_probability_calibration_model(
        self, use_probability_calibration_model: bool) -> LabelWiseProbabilityPredictorConfig:
        """
        Sets whether a model for the calibration of probabilities should be used, if available, or not.

        :param use_probability_calibration_model:   True, if a model for the calibration of probabilities should be
                                                    used, if available, false otherwise
        :return:                                    A `LabelWiseProbabilityPredictorConfig` that allows further
                                                    configuration of the predictor
        """
        self.config_ptr.setUseProbabilityCalibrationModel(use_probability_calibration_model)
        return self


cdef class MarginalizedProbabilityPredictorConfig:
    """
    Allows to configure a predictor that predicts marginalized probabilities for given query examples, which estimate
    the chance of individual labels to be relevant, by summing up the scores that are provided by individual rules of an
    existing rule-based model and comparing the aggregated score vector to the known label vectors according to a
    certain distance measure. The probability for an individual label calculates as the sum of the distances that have
    been obtained for all label vectors, where the respective label is specified to be relevant, divided by the total
    sum of all distances.
    """

    def is_probability_calibration_model_used(self) -> bool:
        """
        Returns whether a model for the calibration of probabilities is used, if available, or not.

        :return: True, if a model for the calibration of probabilities is used, if available, False otherwise
        """
        return self.config_ptr.isProbabilityCalibrationModelUsed()

    def set_use_probability_calibration_model(
        self, use_probability_calibration_model: bool) -> MarginalizedProbabilityPredictorConfig:
        """
        Sets whether a model for the calibration of probabilities should be used, if available, or not.

        :param use_probability_calibration_model:   True, if a model for the calibration of probabilities should be
                                                    used, if available, false otherwise
        :return:                                    A `MarginalizedProbabilityPredictorConfig` that allows further
                                                    configuration of the predictor
        """
        self.config_ptr.setUseProbabilityCalibrationModel(use_probability_calibration_model)
        return self


cdef class ExampleWiseBinaryPredictorConfig:
    """
    Allows to configure a predictor that predicts known label vectors for given query examples by comparing the
    predicted regression scores or probability estimates to the label vectors encountered in the training data.
    """

    def is_based_on_probabilities(self) -> bool:
        """
        Returns whether binary predictions are derived from probability estimates rather than regression scores or not.

        :return: True, if binary predictions are derived from probability estimates rather than regression scores, False
                 otherwise
        """
        return self.config_ptr.isBasedOnProbabilities()

    def set_based_on_probabilities(self, based_on_probabilities: bool) -> ExampleWiseBinaryPredictorConfig:
        """
        Sets whether binary predictions should be derived from probability estimates rather than regression scores or
        not.

        :param based_on_probabilities:  True, if binary predictions should be derived from probability estimates rather
                                        than regression scores, False otherwise
        :return:                        An `ExampleWiseBinaryPredictorConfig` that allows further configuration of the
                                        predictor
        """
        self.config_ptr.setBasedOnProbabilities(based_on_probabilities)
        return self

    def is_probability_calibration_model_used(self) -> bool:
        """
        Returns whether a model for the calibration of probabilities is used, if available, or not.

        :return: True, if a model for the calibration of probabilities is used, if available, False otherwise
        """
        return self.config_ptr.isProbabilityCalibrationModelUsed()

    def set_use_probability_calibration_model(
        self, use_probability_calibration_model: bool) -> ExampleWiseBinaryPredictorConfig:
        """
        Sets whether a model for the calibration of probabilities should be used, if available, or not.

        :param use_probability_calibration_model:   True, if a model for the calibration of probabilities should be
                                                    used, if available, false otherwise
        :return:                                    An `ExampleWiseBinaryPredictorConfig` that allows further
                                                    configuration of the predictor
        """
        self.config_ptr.setUseProbabilityCalibrationModel(use_probability_calibration_model)
        return self


cdef class LabelWiseBinaryPredictorConfig:
    """
    Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
    irrelevant by discretizing the regression scores or probability estimates that are predicted for each label
    individually.
    """

    def is_based_on_probabilities(self) -> bool:
        """
        Returns whether binary predictions are derived from probability estimates rather than regression scores or not.

        :return: True, if binary predictions are derived from probability estimates rather than regression scores, False
                 otherwise
        """
        return self.config_ptr.isBasedOnProbabilities()

    def set_based_on_probabilities(self, based_on_probabilities: bool) -> LabelWiseBinaryPredictorConfig:
        """
        Sets whether binary predictions should be derived from probability estimates rather than regression scores or
        not.

        :param based_on_probabilities:  True, if binary predictions should be derived from probability estimates rather
                                        than regression scores, False otherwise
        :return:                        A `LabelWiseBinaryPredictorConfig` that allows further configuration of the
                                        predictor
        """
        self.config_ptr.setBasedOnProbabilities(based_on_probabilities)
        return self

    def is_probability_calibration_model_used(self) -> bool:
        """
        Returns whether a model for the calibration of probabilities is used, if available, or not.

        :return: True, if a model for the calibration of probabilities is used, if available, False otherwise
        """
        return self.config_ptr.isProbabilityCalibrationModelUsed()

    def set_use_probability_calibration_model(
        self, use_probability_calibration_model: bool) -> LabelWiseBinaryPredictorConfig:
        """
        Sets whether a model for the calibration of probabilities should be used, if available, or not.

        :param use_probability_calibration_model:   True, if a model for the calibration of probabilities should be
                                                    used, if available, false otherwise
        :return:                                    A `LabelWiseBinaryPredictorConfig` that allows further configuration
                                                    of the predictor
        """
        self.config_ptr.setUseProbabilityCalibrationModel(use_probability_calibration_model)
        return self


cdef class GfmBinaryPredictorConfig:
    """
    Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
    irrelevant by discretizing the regression scores or probability estimates that are predicted for each label
    according to the general F-measure maximizer (GFM).
    """

    def is_probability_calibration_model_used(self) -> bool:
        """
        Returns whether a model for the calibration of probabilities is used, if available, or not.

        :return: True, if a model for the calibration of probabilities is used, if available, False otherwise
        """
        return self.config_ptr.isProbabilityCalibrationModelUsed()

    def set_use_probability_calibration_model(self,
                                              use_probability_calibration_model: bool) -> GfmBinaryPredictorConfig:
        """
        Sets whether a model for the calibration of probabilities should be used, if available, or not.

        :param use_probability_calibration_model:   True, if a model for the calibration of probabilities should be
                                                    used, if available, false otherwise
        :return:                                    A `GfmBinaryPredictorConfig` that allows further configuration of
                                                    the predictor
        """
        self.config_ptr.setUseProbabilityCalibrationModel(use_probability_calibration_model)
        return self
