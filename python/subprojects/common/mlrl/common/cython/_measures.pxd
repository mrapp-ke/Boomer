"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef extern from "common/measures/measure_evaluation.hpp" nogil:

    cdef cppclass IEvaluationMeasure:
        pass

        
cdef extern from "common/measures/measure_similarity.hpp" nogil:

    cdef cppclass ISimilarityMeasure:
        pass
