"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from mlrl.boosting.cython._blas cimport init_blas
from mlrl.boosting.cython._lapack cimport init_lapack

from libcpp.memory cimport make_shared
from libcpp.utility cimport move


cdef class ExampleWiseRuleEvaluation:
    """
    A wrapper for the pure virtual C++ class `IExampleWiseRuleEvaluation`.
    """
    pass


cdef class RegularizedExampleWiseRuleEvaluationFactory(ExampleWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `RegularizedExampleWiseRuleEvaluationFactory`.
    """

    def __cinit__(self, float64 l2_regularization_weight):
        """
        :param l2_regularization_weight: The weight of the L2 regularization that is applied for calculating the scores
                                         to be predicted by rules
        """
        cdef shared_ptr[Blas] blas_ptr = <shared_ptr[Blas]>move(init_blas())
        cdef shared_ptr[Lapack] lapack_ptr = <shared_ptr[Lapack]>move(init_lapack())
        self.rule_evaluation_factory_ptr = <shared_ptr[IExampleWiseRuleEvaluationFactory]>make_shared[RegularizedExampleWiseRuleEvaluationFactoryImpl](
            l2_regularization_weight, blas_ptr, lapack_ptr)


cdef class EqualWidthBinningExampleWiseRuleEvaluationFactory(ExampleWiseRuleEvaluationFactory):
    """
    A wrapper for the C++ class `EqualWidthBinningExampleWiseRuleEvaluationFactory`.
    """

    def __cinit__(self, float64 l2_regularization_weight, float32 bin_ratio, uint32 min_bins, uint32 max_bins):
        """
        :param l2_regularization_weight:    The weight of the L2 regularization that is applied for calculating the
                                            scores to be predicted by rules
        :param bin_ratio:                   A percentage that specifies how many bins should be used to assign labels to
        :param min_bins:                    The minimum number of bins to be used to assign labels to
        :param max_bins:                    The maximum number of bins to be used to assign labels to
        """
        cdef shared_ptr[Blas] blas_ptr = <shared_ptr[Blas]>move(init_blas())
        cdef shared_ptr[Lapack] lapack_ptr = <shared_ptr[Lapack]>move(init_lapack())
        self.rule_evaluation_factory_ptr = <shared_ptr[IExampleWiseRuleEvaluationFactory]>make_shared[EqualWidthBinningExampleWiseRuleEvaluationFactoryImpl](
            l2_regularization_weight, bin_ratio, min_bins, max_bins, blas_ptr, lapack_ptr)
