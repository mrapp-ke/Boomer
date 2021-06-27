"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.common.cython._types cimport uint32
from mlrl.boosting.cython.losses_label_wise cimport LabelWiseLoss
from mlrl.boosting.cython.rule_evaluation_label_wise cimport LabelWiseRuleEvaluationFactory

from libcpp.memory cimport make_shared


cdef class LabelWiseStatisticsProviderFactory(StatisticsProviderFactory):
    """
    A wrapper for the C++ class `LabelWiseStatisticsProviderFactory`.
    """

    def __cinit__(self, LabelWiseLoss loss_function, LabelWiseRuleEvaluationFactory default_rule_evaluation_factory,
                  LabelWiseRuleEvaluationFactory rule_evaluation_factory, uint32 num_threads):
        """
        :param loss_function:                   The loss function to be used for calculating gradients and Hessians
        :param default_rule_evaluation_factory: The `LabelWiseRuleEvaluationFactory` that allows to create instances of
                                                the class that should be used for calculating the predictions, as well
                                                as corresponding quality scores, of the default rule
        :param rule_evaluation:                 The `LabelWiseRuleEvaluationFactory` that allows to create instances of
                                                the class that should be used for calculating the predictions, as well
                                                as corresponding quality scores, of rules
        :param num_threads:                     The number of CPU threads to be used to calculate the initial statistics
                                                in parallel. Must be at least 1
        """
        self.statistics_provider_factory_ptr = <shared_ptr[IStatisticsProviderFactory]>make_shared[LabelWiseStatisticsProviderFactoryImpl](
            loss_function.loss_function_ptr, default_rule_evaluation_factory.rule_evaluation_factory_ptr,
            rule_evaluation_factory.rule_evaluation_factory_ptr, num_threads)
