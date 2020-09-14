from boomer.common._arrays cimport float64
from boomer.boosting._blas cimport Blas
from boomer.boosting._lapack cimport Lapack

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/example_wise_rule_evaluation.h" namespace "boosting" nogil:

    cdef cppclass AbstractExampleWiseRuleEvaluation:
        pass


    cdef cppclass RegularizedExampleWiseRuleEvaluationImpl(AbstractExampleWiseRuleEvaluation):

        # Constructors:

        RegularizedExampleWiseRuleEvaluationImpl(float64 l2RegularizationWeight, shared_ptr[Blas] blasPtr,
                                                 shared_ptr[Lapack] lapackPtr) except +


cdef class ExampleWiseRuleEvaluation:

    # Attributes:

    cdef shared_ptr[AbstractExampleWiseRuleEvaluation] rule_evaluation_ptr


cdef class RegularizedExampleWiseRuleEvaluation(ExampleWiseRuleEvaluation):
    pass
