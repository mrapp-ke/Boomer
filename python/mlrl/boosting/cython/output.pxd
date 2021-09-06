from mlrl.common.cython._types cimport uint8, uint32, float64
from mlrl.common.cython._measures cimport ISimilarityMeasure
from mlrl.common.cython.measures cimport SimilarityMeasure
from mlrl.common.cython.output cimport AbstractBinaryPredictor, AbstractNumericalPredictor, IPredictor, ISparsePredictor

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/output/predictor_probability_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseTransformationFunction:
        pass


    cdef cppclass LogisticFunctionImpl"boosting::LogisticFunction"(ILabelWiseTransformationFunction):
        pass


    cdef cppclass LabelWiseProbabilityPredictorImpl"boosting::LabelWiseProbabilityPredictor"(IPredictor[float64]):

        # Constructors:

        LabelWiseProbabilityPredictorImpl(unique_ptr[ILabelWiseTransformationFunction] transformationFunctionPtr,
                                          uint32 numThreads) except +


cdef extern from "boosting/output/predictor_regression_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseRegressionPredictorImpl"boosting::LabelWiseRegressionPredictor"(IPredictor[float64]):

        # Constructors:

        LabelWiseRegressionPredictorImpl(uint32 numThreads) except +


cdef extern from "boosting/output/predictor_classification_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseClassificationPredictorImpl"boosting::LabelWiseClassificationPredictor"(
            ISparsePredictor[uint8]):

        # Constructors:

        LabelWiseClassificationPredictorImpl(float64 threshold, uint32 numThreads) except +


cdef extern from "boosting/output/predictor_classification_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ExampleWiseClassificationPredictorImpl"boosting::ExampleWiseClassificationPredictor"(
            ISparsePredictor[uint8]):

        # Constructors:

        ExampleWiseClassificationPredictorImpl(unique_ptr[ISimilarityMeasure] measurePtr, uint32 numThreads) except +


cdef class LabelWiseTransformationFunction:

    # Attributes:

    cdef unique_ptr[ILabelWiseTransformationFunction] transformation_function_ptr


cdef class LogisticFunction(LabelWiseTransformationFunction):
    pass


cdef class LabelWiseProbabilityPredictor(AbstractNumericalPredictor):

    # Attributes:

    cdef LabelWiseTransformationFunction transformation_function

    cdef uint32 num_threads


cdef class LabelWiseRegressionPredictor(AbstractNumericalPredictor):

    # Attributes

    cdef uint32 num_threads


cdef class LabelWiseClassificationPredictor(AbstractBinaryPredictor):

    # Attributes:

    cdef float64 threshold

    cdef uint32 num_threads


cdef class ExampleWiseClassificationPredictor(AbstractBinaryPredictor):

    # Attributes

    cdef SimilarityMeasure measure

    cdef uint32 num_threads
