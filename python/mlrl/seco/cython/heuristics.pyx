"""
@author Jakob Steeg (jakob.steeg@gmail.com)
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
@author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
"""
from libcpp.memory cimport make_unique


cdef class Heuristic:
    """
    A wrapper for the pure virtual C++ class `IHeuristic`.
    """
    pass


cdef class Accuracy(Heuristic):
    """
    A wrapper for the C++ class `Accuracy`.
    """

    def __cinit__(self):
        self.heuristic_ptr = <unique_ptr[IHeuristic]>make_unique[AccuracyImpl]()


cdef class Precision(Heuristic):
    """
    A wrapper for the C++ class `Precision`.
    """

    def __cinit__(self):
        self.heuristic_ptr = <unique_ptr[IHeuristic]>make_unique[PrecisionImpl]()


cdef class Recall(Heuristic):
    """
    A wrapper for the C++ class `Recall`.
    """

    def __cinit__(self):
        self.heuristic_ptr = <unique_ptr[IHeuristic]>make_unique[RecallImpl]()


cdef class Laplace(Heuristic):
    """
    A wrapper for the C++ class 'Laplace'.
    """

    def __cinit__(self):
        self.heuristic_ptr = <unique_ptr[IHeuristic]>make_unique[LaplaceImpl]()


cdef class WRA(Heuristic):
    """
    A wrapper for the C++ class `WRA`.
    """

    def __cinit__(self):
        self.heuristic_ptr = <unique_ptr[IHeuristic]>make_unique[WRAImpl]()


cdef class FMeasure(Heuristic):
    """
    A wrapper for the C++ class `FMeasure`.
    """

    def __cinit__(self, float64 beta):
        """
        :param beta: The value of the beta-parameter. Must be at least 0
        """
        self.heuristic_ptr = <unique_ptr[IHeuristic]>make_unique[FMeasureImpl](beta)


cdef class MEstimate(Heuristic):
    """
    A wrapper for the C++ class `MEstimate`.
    """

    def __cinit__(self, float64 m):
        """
        :param m: The value of the m-parameter. Must be at least 0
        """
        self.heuristic_ptr = <unique_ptr[IHeuristic]>make_unique[MEstimateImpl](m)
