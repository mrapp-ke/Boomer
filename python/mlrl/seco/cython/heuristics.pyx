"""
@author Jakob Steeg (jakob.steeg@gmail.com)
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_shared


cdef class Heuristic:
    """
    A wrapper for the pure virtual C++ class `IHeuristic`.
    """
    pass


cdef class Precision(Heuristic):
    """
    A wrapper for the C++ class `Precision`.
    """

    def __cinit__(self):
        self.heuristic_ptr = <shared_ptr[IHeuristic]>make_shared[PrecisionImpl]()


cdef class Recall(Heuristic):
    """
    A wrapper for the C++ class `Recall`.
    """

    def __cinit__(self):
        self.heuristic_ptr = <shared_ptr[IHeuristic]>make_shared[RecallImpl]()


cdef class WRA(Heuristic):
    """
    A wrapper for the C++ class `WRA`.
    """

    def __cinit__(self):
        self.heuristic_ptr = <shared_ptr[IHeuristic]>make_shared[WRAImpl]()


cdef class HammingLoss(Heuristic):
    """
    A wrapper for the C++ class `HammingLoss`.
    """

    def __cinit__(self):
        self.heuristic_ptr = <shared_ptr[IHeuristic]>make_shared[HammingLossImpl]()


cdef class FMeasure(Heuristic):
    """
    A wrapper for the C++ class `FMeasure`.
    """

    def __cinit__(self, float64 beta = 0.5):
        """
        :param beta: The value of the beta-parameter. Must be at least 0
        """
        self.heuristic_ptr = <shared_ptr[IHeuristic]>make_shared[FMeasureImpl](beta)


cdef class MEstimate(Heuristic):
    """
    A wrapper for the C++ class `MEstimate`.
    """

    def __cinit__(self, float64 m = 22.466):
        """
        :param m: The value of the m-parameter. Must be at least 0
        """
        self.heuristic_ptr = <shared_ptr[IHeuristic]>make_shared[MEstimateImpl](m)
