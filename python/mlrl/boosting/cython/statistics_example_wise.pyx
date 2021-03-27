"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.common.cython._types cimport uint32
from mlrl.boosting.cython.losses_example_wise cimport ExampleWiseLoss
from mlrl.boosting.cython.rule_evaluation_example_wise cimport ExampleWiseRuleEvaluationFactory

from libcpp.memory cimport make_shared


cdef class ExampleWiseStatisticsProviderFactory(StatisticsProviderFactory):
    """
    A wrapper for the C++ class `ExampleWiseStatisticsProviderFactory`.
    """

    def __cinit__(self, ExampleWiseLoss loss_function, ExampleWiseRuleEvaluationFactory default_rule_evaluation_factory,
                  ExampleWiseRuleEvaluationFactory rule_evaluation_factory, uint32 num_threads):
        """
        :param loss_function:                   The loss function to be used for calculating gradients and Hessians
        :param default_rule_evaluation_factory: The `ExampleWiseRuleEvaluation` to be used for calculating the
                                                predictions, as well as corresponding quality scores, of the default
                                                rule
        :param rule_evaluation_factory:         The `ExampleWiseRuleEvaluationFactory` to be used for calculating the
                                                predictions, as well as corresponding quality scores, of rules
        :param label_matrix:                    A label matrix that provides random access to the labels of the training
                                                examples
        :param num_threads:                     The number of CPU threads to be used to calculate the initial statistics
                                                in parallel. Must be at least 1
        """
        self.statistics_provider_factory_ptr = <shared_ptr[IStatisticsProviderFactory]>make_shared[ExampleWiseStatisticsProviderFactoryImpl](
            loss_function.loss_function_ptr, default_rule_evaluation_factory.rule_evaluation_factory_ptr,
            rule_evaluation_factory.rule_evaluation_factory_ptr, num_threads)
