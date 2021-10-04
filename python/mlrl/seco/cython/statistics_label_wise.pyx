"""
@author Jakob Steeg (jakob.steeg@gmail.com)
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.seco.cython.rule_evaluation_label_wise cimport LabelWiseRuleEvaluationFactory

from libcpp.utility cimport move
from libcpp.memory cimport unique_ptr, make_unique


cdef class DenseLabelWiseStatisticsProviderFactory(StatisticsProviderFactory):
    """
    A wrapper for the C++ class `LabelWiseStatisticsProviderFactory`.
    """

    def __cinit__(self, LabelWiseRuleEvaluationFactory default_rule_evaluation_factory not None,
                  LabelWiseRuleEvaluationFactory regular_rule_evaluation_factory not None,
                  LabelWiseRuleEvaluationFactory pruning_rule_evaluation_factory not None):
        """
        :param default_rule_evaluation_factory: The `LabelWiseRuleEvaluationFactory` that allows to create instances of
                                                the class that should be used for calculating the predictions, as well
                                                as corresponding quality scores, of the default rule
        :param regular_rule_evaluation_factory: The `LabelWiseRuleEvaluation` that allows to create instances of the
                                                class that should be used for calculating the predictions, as well as
                                                corresponding quality scores, of all remaining rules
        :param pruning_rule_evaluation_factory: The `LabelWiseRuleEvaluation` that allows to create instances of the
                                                class that should be used for calculating the predictions, as well as
                                                corresponding quality scores, when pruning rules
        """
        self.statistics_provider_factory_ptr = <unique_ptr[IStatisticsProviderFactory]>make_unique[DenseLabelWiseStatisticsProviderFactoryImpl](
            move(default_rule_evaluation_factory.rule_evaluation_factory_ptr),
            move(regular_rule_evaluation_factory.rule_evaluation_factory_ptr),
            move(pruning_rule_evaluation_factory.rule_evaluation_factory_ptr))
