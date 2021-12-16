"""
@author Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.boosting.cython._blas cimport init_blas
from mlrl.boosting.cython._lapack cimport init_lapack
from mlrl.boosting.cython.label_binning cimport LabelBinningFactory

from libcpp.memory cimport make_unique
from libcpp.utility cimport move


cdef class ExampleWiseRuleEvaluationFactory:
    """
    A wrapper for the pure virtual C++ class `IExampleWiseRuleEvaluationFactory`.
    """
    pass


cdef class ExampleWiseCompleteRuleEvaluationFactory(ExampleWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `ExampleWiseCompleteRuleEvaluationFactory`.
    """

    def __cinit__(self, float64 l1_regularization_weight, float64 l2_regularization_weight):
        """
        :param l1_regularization_weight:    The weight of the L1 regularization that is applied for calculating the
                                            scores to be predicted by rules
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by rules
        """
        self.rule_evaluation_factory_ptr = <unique_ptr[IExampleWiseRuleEvaluationFactory]>make_unique[ExampleWiseCompleteRuleEvaluationFactoryImpl](
            l1_regularization_weight, l2_regularization_weight, move(init_blas()), move(init_lapack()))


cdef class ExampleWiseCompleteBinnedRuleEvaluationFactory(ExampleWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `ExampleWiseCompleteBinnedRuleEvaluationFactory`.
    """

    def __cinit__(self, float64 l1_regularization_weight, float64 l2_regularization_weight,
                  LabelBinningFactory label_binning_factory not None):
        """
        :param l1_regularization_weight:    The weight of the L1 regularization that is applied for calculating the
                                            scores to be predicted by rules
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by rules
        :param label_binning_factory:       A `LabelBinningFactory` that allows to create the implementation that should
                                            be used to assign labels to bins
        """
        self.rule_evaluation_factory_ptr = <unique_ptr[IExampleWiseRuleEvaluationFactory]>make_unique[ExampleWiseCompleteBinnedRuleEvaluationFactoryImpl](
            l1_regularization_weight, l2_regularization_weight, move(label_binning_factory.label_binning_factory_ptr),
            move(init_blas()), move(init_lapack()))
