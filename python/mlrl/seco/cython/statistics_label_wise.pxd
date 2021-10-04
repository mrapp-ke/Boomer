from mlrl.common.cython.statistics cimport StatisticsProviderFactory, IStatisticsProviderFactory
from mlrl.seco.cython.rule_evaluation_label_wise cimport ILabelWiseRuleEvaluationFactory

from libcpp.memory cimport unique_ptr


cdef extern from "seco/statistics/statistics_provider_factory_label_wise_dense.hpp" namespace "seco" nogil:

    cdef cppclass DenseLabelWiseStatisticsProviderFactoryImpl"seco::DenseLabelWiseStatisticsProviderFactory"(
            IStatisticsProviderFactory):

        # Constructors:

        DenseLabelWiseStatisticsProviderFactoryImpl(
            unique_ptr[ILabelWiseRuleEvaluationFactory] defaultRuleEvaluationFactoryPtr,
            unique_ptr[ILabelWiseRuleEvaluationFactory] regularRuleEvaluationFactoryPtr,
            unique_ptr[ILabelWiseRuleEvaluationFactory] pruningRuleEvaluationFactoryPtr) except +


cdef class DenseLabelWiseStatisticsProviderFactory(StatisticsProviderFactory):
    pass
