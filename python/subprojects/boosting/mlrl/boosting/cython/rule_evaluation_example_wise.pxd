from mlrl.common.cython._types cimport uint32, float32, float64
from mlrl.boosting.cython._blas cimport Blas
from mlrl.boosting.cython._lapack cimport Lapack
from mlrl.boosting.cython.label_binning cimport ILabelBinningFactory

from libcpp.memory cimport unique_ptr


cdef extern from "boosting/rule_evaluation/rule_evaluation_example_wise.hpp" namespace "boosting" nogil:

    cdef cppclass IExampleWiseRuleEvaluationFactory:
        pass


cdef extern from "boosting/rule_evaluation/rule_evaluation_example_wise_complete.hpp" namespace "boosting" nogil:

    cdef cppclass ExampleWiseCompleteRuleEvaluationFactoryImpl"boosting::ExampleWiseCompleteRuleEvaluationFactory"(
            IExampleWiseRuleEvaluationFactory):

        # Constructors:

        ExampleWiseCompleteRuleEvaluationFactoryImpl(float64 l1RegularizationWeight, float64 l2RegularizationWeight,
                                                     unique_ptr[Blas] blasPtr, unique_ptr[Lapack] lapackPtr) except +


cdef extern from "boosting/rule_evaluation/rule_evaluation_example_wise_complete_binned.hpp" namespace "boosting" nogil:

    cdef cppclass ExampleWiseCompleteBinnedRuleEvaluationFactoryImpl"boosting::ExampleWiseCompleteBinnedRuleEvaluationFactory"(
            IExampleWiseRuleEvaluationFactory):

        # Constructors:

        ExampleWiseCompleteBinnedRuleEvaluationFactoryImpl(float64 l1RegularizationWeight,
                                                           float64 l2RegularizationWeight,
                                                           unique_ptr[ILabelBinningFactory] labelBinningFactoryPtr,
                                                           unique_ptr[Blas] blasPtr,
                                                           unique_ptr[Lapack] lapackPtr) except +


cdef class ExampleWiseRuleEvaluationFactory:

    # Attributes:

    cdef unique_ptr[IExampleWiseRuleEvaluationFactory] rule_evaluation_factory_ptr


cdef class ExampleWiseCompleteRuleEvaluationFactory(ExampleWiseRuleEvaluationFactory):
    pass


cdef class ExampleWiseCompleteBinnedRuleEvaluationFactory(ExampleWiseRuleEvaluationFactory):
    pass
