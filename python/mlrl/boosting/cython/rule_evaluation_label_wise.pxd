from mlrl.common.cython._types cimport uint32, float32, float64
from mlrl.boosting.cython.label_binning cimport ILabelBinningFactory

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/rule_evaluation/rule_evaluation_label_wise.hpp" namespace "boosting" nogil:

    cdef cppclass ILabelWiseRuleEvaluationFactory:
        pass


cdef extern from "boosting/rule_evaluation/rule_evaluation_label_wise_single.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseSingleLabelRuleEvaluationFactoryImpl"boosting::LabelWiseSingleLabelRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):

        # Constructors:

        LabelWiseSingleLabelRuleEvaluationFactoryImpl(float64 l2RegularizationWeight) except +


cdef extern from "boosting/rule_evaluation/rule_evaluation_label_wise_complete.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseCompleteRuleEvaluationFactoryImpl"boosting::LabelWiseCompleteRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):

        # Constructors:

        LabelWiseCompleteRuleEvaluationFactoryImpl(float64 l2RegularizationWeight) except +


cdef extern from "boosting/rule_evaluation/rule_evaluation_label_wise_complete_binned.hpp" namespace "boosting" nogil:

    cdef cppclass LabelWiseCompleteBinnedRuleEvaluationFactoryImpl"boosting::LabelWiseCompleteBinnedRuleEvaluationFactory"(
            ILabelWiseRuleEvaluationFactory):

        # Constructors:

        LabelWiseCompleteBinnedRuleEvaluationFactoryImpl(
            float64 l2RegularizationWeight, unique_ptr[ILabelBinningFactory] labelBinningFactoryPtr) except +


cdef class LabelWiseRuleEvaluationFactory:

    # Attributes:

    cdef unique_ptr[ILabelWiseRuleEvaluationFactory] rule_evaluation_factory_ptr


cdef class LabelWiseSingleLabelRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass


cdef class LabelWiseCompleteRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass


cdef class LabelWiseCompleteBinnedRuleEvaluationFactory(LabelWiseRuleEvaluationFactory):
    pass
