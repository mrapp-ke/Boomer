"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class SimilarityMeasure:
    """
    A wrapper for the pure virtual C++ class `ISimilarityMeasure`.
    """

    cdef unique_ptr[ISimilarityMeasure] get_similarity_measure_ptr(self):
        pass


cdef class EvaluationMeasure(SimilarityMeasure):
    """
    A wrapper for the pure virtual C++ class `IEvaluationMeasure`.
    """

    cdef unique_ptr[IEvaluationMeasure] get_evaluation_measure_ptr(self):
        pass
