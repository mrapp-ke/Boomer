"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
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
