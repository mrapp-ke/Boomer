from mlrl.common.cython._types cimport float64

from libcpp.memory cimport shared_ptr


cdef extern from "boosting/rule_evaluation/rule_evaluation_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseRuleEvaluationFactory:
        pass


cdef extern from "boosting/rule_evaluation/rule_evaluation_label_wise_regularized.hpp" namespace "boosting" nogil:

    cdef cppclass RegularizedLabelWiseRuleEvaluationFactoryImpl"boosting::RegularizedLabelWiseRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):

        # Constructors:

        RegularizedLabelWiseRuleEvaluationFactoryImpl(float64 l2RegularizationWeight) except +


cdef class LabelWiseRuleEvaluationFactory:

    # Attributes:

    cdef shared_ptr[ILabelWiseRuleEvaluationFactory] rule_evaluation_factory_ptr


cdef class RegularizedLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass
