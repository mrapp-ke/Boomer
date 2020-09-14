from boomer.common._arrays cimport float64

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/label_wise_rule_evaluation.h" namespace "boosting" nogil:

    cdef cppclass AbstractLabelWiseRuleEvaluation:
        pass


    cdef cppclass RegularizedLabelWiseRuleEvaluationImpl(AbstractLabelWiseRuleEvaluation):

        # Constructors:

        RegularizedLabelWiseRuleEvaluationImpl(float64 l2RegularizationWeight) except +


cdef class LabelWiseRuleEvaluation:

    # Attributes:

    cdef shared_ptr[AbstractLabelWiseRuleEvaluation] rule_evaluation_ptr


cdef class RegularizedLabelWiseRuleEvaluation(LabelWiseRuleEvaluation):
    pass
