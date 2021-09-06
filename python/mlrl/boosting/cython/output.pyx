"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.common.cython._types cimport uint32

from libcpp.utility cimport move
from libcpp.memory cimport unique_ptr, make_unique


cdef class LabelWiseTransformationFunction:
    """
    A wrapper for the pure virtual C++ class `ILabelWiseTransformationFunction`.
    """
    pass


cdef class LogisticFunction(LabelWiseTransformationFunction):
    """
    A wrapper for the C++ class `LogisticFunction`.
    """

    def __cinit__(self):
        self.transformation_function_ptr = <unique_ptr[ILabelWiseTransformationFunction]>make_unique[LogisticFunctionImpl]()

    def __reduce__(self):
        return (LogisticFunction, ())


cdef class LabelWiseProbabilityPredictor(AbstractNumericalPredictor):

    def __cinit__(self, uint32 num_labels, LabelWiseTransformationFunction transformation_function not None,
                  uint32 num_threads):
        self.num_labels = num_labels
        self.transformation_function = transformation_function
        self.num_threads = num_threads
        self.predictor_ptr = <unique_ptr[IPredictor[float64]]>make_unique[LabelWiseProbabilityPredictorImpl](
            move(transformation_function.transformation_function_ptr), num_threads)

    def __reduce__(self):
        return (LabelWiseProbabilityPredictor, (self.num_labels, self.transformation_function, self.num_threads))


cdef class LabelWiseRegressionPredictor(AbstractNumericalPredictor):
    """
    A wrapper for the C++ class `LabelWiseRegressionPredictor`.
    """

    def __cinit__(self, uint32 num_labels, uint32 num_threads):
        """
        :param num_labels:  The total number of available labels
        :param num_threads: The number of CPU threads to be used to make predictions for different query examples in
                            parallel. Must be at least 1
        """
        self.num_labels = num_labels
        self.num_threads = num_threads
        self.predictor_ptr = <unique_ptr[IPredictor[float64]]>make_unique[LabelWiseRegressionPredictorImpl](num_threads)

    def __reduce__(self):
        return (LabelWiseRegressionPredictor, (self.num_labels, self.num_threads))


cdef class LabelWiseClassificationPredictor(AbstractBinaryPredictor):
    """
    A wrapper for the C++ class `LabelWiseClassificationPredictor`.
    """

    def __cinit__(self, uint32 num_labels, float64 threshold, uint32 num_threads):
        """
        :param num_labels:  The total number of available labels
        :param thresholds:  The threshold to be used for making predictions
        :param num_threads: The number of CPU threads to be used to make predictions for different query examples in
                            parallel. Must be at least 1
        """
        self.num_labels = num_labels
        self.threshold = threshold
        self.num_threads = num_threads
        self.predictor_ptr = <unique_ptr[ISparsePredictor[uint8]]>make_unique[LabelWiseClassificationPredictorImpl](
            threshold, num_threads)

    def __reduce__(self):
        return (LabelWiseClassificationPredictor, (self.num_labels, self.threshold, self.num_threads))


cdef class ExampleWiseClassificationPredictor(AbstractBinaryPredictor):
    """
    A wrapper for the C++ class `ExampleWiseClassificationPredictor`.
    """

    def __cinit__(self, uint32 num_labels, SimilarityMeasure measure not None, uint32 num_threads):
        """
        :param num_labels:  The total number of available labels
        :param measure:     The measure to be used
        :param num_threads: The number of CPU threads to be used to make predictions for different query examples in
                            parallel. Must be at least 1
        """
        self.num_labels = num_labels
        self.measure = measure
        self.num_threads = num_threads
        cdef unique_ptr[ISimilarityMeasure] measure_ptr = measure.get_similarity_measure_ptr()
        self.predictor_ptr = <unique_ptr[ISparsePredictor[uint8]]>make_unique[ExampleWiseClassificationPredictorImpl](
            move(measure_ptr), num_threads)

    def __reduce__(self):
        return (ExampleWiseClassificationPredictor, (self.num_labels, self.measure, self.num_threads))
