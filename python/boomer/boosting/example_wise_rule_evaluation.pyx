"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to calculate the predictions of rules, as well as corresponding quality scores, such that
they minimize a loss function that is applied example-wise.
"""
from boomer.boosting._blas cimport init_blas
from boomer.boosting._lapack cimport init_lapack

from libcpp.memory cimport make_shared


cdef class ExampleWiseRuleEvaluation:
    """
    A wrapper for the abstract C++ class `AbstractExampleWiseRuleEvaluation`.
    """
    pass


cdef class RegularizedExampleWiseRuleEvaluation(ExampleWiseRuleEvaluation):
    """
    A wrapper for the C++ class `RegularizedExampleWiseRuleEvaluationImpl`.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        cdef shared_ptr[Blas] blas_ptr
        blas_ptr.reset(init_blas())
        cdef shared_ptr[Lapack] lapack_ptr
        lapack_ptr.reset(init_lapack())
        self.rule_evaluation_ptr = <shared_ptr[AbstractExampleWiseRuleEvaluation]>make_shared[RegularizedExampleWiseRuleEvaluationImpl](
            l2_regularization_weight, blas_ptr, lapack_ptr)
