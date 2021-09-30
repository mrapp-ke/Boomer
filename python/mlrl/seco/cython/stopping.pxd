from mlrl.common.cython._types cimport float64
from mlrl.common.cython.stopping cimport IStoppingCriterion, StoppingCriterion


cdef extern from "seco/stopping/stopping_criterion_coverage.hpp" nogil:

    cdef cppclass CoverageStoppingCriterionImpl"seco::CoverageStoppingCriterion"(IStoppingCriterion):

        # Constructors:

        CoverageStoppingCriterionImpl(float64 threshold) except +


cdef class CoverageStoppingCriterion(StoppingCriterion):
    pass
