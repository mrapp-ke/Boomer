from mlrl.common.cython._measures cimport IEvaluationMeasure, ISimilarityMeasure

from libcpp.memory cimport unique_ptr


cdef class SimilarityMeasure:

    # Functions:

    cdef unique_ptr[ISimilarityMeasure] get_similarity_measure_ptr(self)


cdef class EvaluationMeasure(SimilarityMeasure):

    # Functions:

    cdef unique_ptr[IEvaluationMeasure] get_evaluation_measure_ptr(self)
