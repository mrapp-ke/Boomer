from mlrl.common.cython._measures cimport IEvaluationMeasure, ISimilarityMeasure

from libcpp.memory cimport shared_ptr


cdef class SimilarityMeasure:

    # Functions:

    cdef shared_ptr[ISimilarityMeasure] get_similarity_measure_ptr(self)


cdef class EvaluationMeasure(SimilarityMeasure):

    # Functions:

    cdef shared_ptr[IEvaluationMeasure] get_evaluation_measure_ptr(self)
