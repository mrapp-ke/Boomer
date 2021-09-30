"""
@author Jakob Steeg (jakob.steeg@gmail.com)
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.seco.cython.heuristics cimport Heuristic

from libcpp.utility cimport move
from libcpp.memory cimport make_unique


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
        self.lift_function_ptr = <unique_ptr[ILiftFunction]>make_unique[PeakLiftFunctionImpl](num_labels, peak_label,
                                                                                              max_lift, curvature)


cdef class LabelWiseRuleEvaluationFactory:
    """
    A wrapper for the pure virtual C++ class `ILabelWiseRuleEvaluationFactory`.
    """
    pass


cdef class LabelWiseMajorityRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `LabelWiseMajorityRuleEvaluationFactory`.
    """

    def __cinit__(self):
        self.rule_evaluation_factory_ptr = <unique_ptr[ILabelWiseRuleEvaluationFactory]>make_unique[LabelWiseMajorityRuleEvaluationFactoryImpl]()


cdef class LabelWisePartialRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `LabelWisePartialRuleEvaluationFactory`.
    """

    def __cinit__(self, Heuristic heuristic not None, LiftFunction lift_function not None):
        """
        :param heuristic:       The heuristic that should be used
        :param liftFunction:    The lift function that should be used
        """
        self.rule_evaluation_factory_ptr = <unique_ptr[ILabelWiseRuleEvaluationFactory]>make_unique[LabelWisePartialRuleEvaluationFactoryImpl](
            move(heuristic.heuristic_ptr), move(lift_function.lift_function_ptr))


cdef class LabelWiseSingleLabelRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `LabelWiseSingleLabelRuleEvaluationFactory`.
    """

    def __cinit__(self, Heuristic heuristic not None):
        """
        :param heuristic: The heuristic that should be used
        """
        self.rule_evaluation_factory_ptr = <unique_ptr[ILabelWiseRuleEvaluationFactory]>make_unique[LabelWiseSingleLabelRuleEvaluationFactoryImpl](
            move(heuristic.heuristic_ptr))
