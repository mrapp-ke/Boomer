"""
@author Jakob Steeg (jakob.steeg@gmail.com)
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_shared


cdef class LiftFunction:
    """
    A wrapper for the pure virtual C++ class `ILiftFunction`.
    """
    pass


cdef class PeakLiftFunction(LiftFunction):
    """
    A wrapper for the C++ class `PeakLiftFunction`.
    """

    def __cinit__(self, uint32 num_labels, uint32 peak_label, float64 max_lift, float64 curvature):
        """
        :param num_labels:  The total number of available labels. Must be greater than 0
        :param peak_label:  The number of labels for which the lift is maximum. Must be in [1, numLabels]
        :param max_lift:    The lift at the peak label. Must be at least 1
        :param curvature:   The curvature of the lift function. A greater value results in a steeper curvature, a
                            smaller value results in a flatter curvature. Must be greater than 0
        """
        self.lift_function_ptr = <shared_ptr[ILiftFunction]>make_shared[PeakLiftFunctionImpl](num_labels, peak_label,
                                                                                              max_lift, curvature)


cdef class PartialHeadRefinementFactory(HeadRefinementFactory):
    """
    A wrapper for the C++ class `PartialHeadRefinementFactory`.
    """

    def __cinit__(self, LiftFunction lift_function):
        self.head_refinement_factory_ptr = <shared_ptr[IHeadRefinementFactory]>make_shared[PartialHeadRefinementFactoryImpl](
            lift_function.lift_function_ptr)
