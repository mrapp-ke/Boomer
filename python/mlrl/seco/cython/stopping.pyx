"""
@author Jakob Steeg (jakob.steeg@gmail.com)
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport shared_ptr, make_shared


cdef class CoverageStoppingCriterion(StoppingCriterion):
    """
    A wrapper for the C++ class `CoverageStoppingCriterion`.
    """

    def __cinit__(self, float64 threshold):
        """
        :param threshold: The threshold
        """
        self.stopping_criterion_ptr = <shared_ptr[IStoppingCriterion]>make_shared[CoverageStoppingCriterionImpl](
            threshold)
