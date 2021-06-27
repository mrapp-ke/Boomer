"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from libcpp.memory cimport make_shared


cdef class LabelWiseRuleEvaluationFactory:
    """
    A wrapper for the pure virtual C++ class `ILabelWiseRuleEvaluationFactory`.
    """
    pass


cdef class RegularizedLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `RegularizedLabelWiseRuleEvaluationFactory`.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        self.rule_evaluation_factory_ptr = <shared_ptr[ILabelWiseRuleEvaluationFactory]>make_shared[RegularizedLabelWiseRuleEvaluationFactoryImpl](
            l2_regularization_weight)


cdef class EqualWidthBinningLabelWiseRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `EqualWidthBinningLabelWiseRuleEvaluationFactory`.
    """

    def __cinit__(self, float64 l2_regularization_weight, float32 bin_ratio, uint32 min_bins, uint32 max_bins):
        """
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by rules
        :param bin_ratio:                   A percentage that specifies how many bins should be used to assign labels to
        :param min_bins:                    The minimum number of bins to be used to assign labels to
        :param max_bins:                    The maximum number of bins to be used to assign labels to
        """
        self.rule_evaluation_factory_ptr = <shared_ptr[ILabelWiseRuleEvaluationFactory]>make_shared[EqualWidthBinningLabelWiseRuleEvaluationFactoryImpl](
            l2_regularization_weight, bin_ratio, min_bins, max_bins)
