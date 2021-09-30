from mlrl.common.cython._types cimport uint32, float64
from mlrl.seco.cython.heuristics cimport IHeuristic

from libcpp.memory cimport unique_ptr


cdef extern from "seco/rule_evaluation/lift_function.hpp" namespace "seco" nogil:

    cdef cppclass ILiftFunction:
        pass


cdef extern from "seco/rule_evaluation/lift_function_peak.hpp" namespace "seco" nogil:

    cdef cppclass PeakLiftFunctionImpl"seco::PeakLiftFunction"(ILiftFunction):

        # Constructors:

        PeakLiftFunctionImpl(uint32 numLabels, uint32 peakLabel, float64 maxLift, float64 curvature) except +


cdef extern from "seco/rule_evaluation/rule_evaluation_label_wise.hpp" namespace "seco" nogil:

    cdef cppclass ILabelWiseRuleEvaluationFactory:
        pass


cdef extern from "seco/rule_evaluation/rule_evaluation_label_wise_majority.hpp" namespace "seco" nogil:

    cdef cppclass LabelWiseMajorityRuleEvaluationFactoryImpl"seco::LabelWiseMajorityRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):
        pass


cdef extern from "seco/rule_evaluation/rule_evaluation_label_wise_partial.hpp" namespace "seco" nogil:

    cdef cppclass LabelWisePartialRuleEvaluationFactoryImpl"seco::LabelWisePartialRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):

        # Constructors:

        LabelWisePartialRuleEvaluationFactoryImpl(unique_ptr[IHeuristic] heuristicPtr,
                                                  unique_ptr[ILiftFunction] liftFunctionPtr) except +


cdef extern from "seco/rule_evaluation/rule_evaluation_label_wise_single.hpp" namespace "seco" nogil:

    cdef cppclass LabelWiseSingleLabelRuleEvaluationFactoryImpl"seco::LabelWiseSingleLabelRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):

        # Constructors:

        LabelWiseSingleLabelRuleEvaluationFactoryImpl(unique_ptr[IHeuristic] heuristicPtr) except +


cdef class LiftFunction:

    # Attributes:

    cdef unique_ptr[ILiftFunction] lift_function_ptr


cdef class PeakLiftFunction(LiftFunction):
    pass


cdef class LabelWiseRuleEvaluationFactory:

    # Attributes:

    cdef unique_ptr[ILabelWiseRuleEvaluationFactory] rule_evaluation_factory_ptr


cdef class LabelWiseMajorityRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass


cdef class LabelWisePartialRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass


cdef class LabelWiseSingleLabelRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass
