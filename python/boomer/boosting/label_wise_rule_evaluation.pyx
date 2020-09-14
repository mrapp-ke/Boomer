"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides Cython wrappers for C++ classes that allow to calculate the predictions of rules, as well as corresponding
quality scores.
"""
from libcpp.memory cimport make_shared


cdef class LabelWiseRuleEvaluation:
    """
    A wrapper for the abstract C++ class `AbstractLabelWiseRuleEvaluation`.
    """
    pass


cdef class RegularizedLabelWiseRuleEvaluation(LabelWiseRuleEvaluation):
    """
    A wrapper for the C++ class `RegularizedLabelWiseRuleEvaluationImpl`.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        self.rule_evaluation_ptr = <shared_ptr[AbstractLabelWiseRuleEvaluation]>make_shared[RegularizedLabelWiseRuleEvaluationImpl](
            l2_regularization_weight)
