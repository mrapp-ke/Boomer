from boomer.common._arrays cimport uint32, float64
from boomer.common.input_data cimport LabelMatrix, AbstractRandomAccessLabelMatrix
from boomer.common.statistics cimport StatisticsProvider, StatisticsProviderFactory, AbstractStatistics, \
    AbstractRefinementSearch
from boomer.boosting._lapack cimport Lapack
from boomer.boosting.statistics cimport AbstractGradientStatistics
from boomer.boosting.example_wise_losses cimport ExampleWiseLoss, AbstractExampleWiseLoss
from boomer.boosting.example_wise_rule_evaluation cimport ExampleWiseRuleEvaluation, AbstractExampleWiseRuleEvaluation

from libcpp.memory cimport shared_ptr


cdef extern from "cpp/example_wise_statistics.h" namespace "boosting" nogil:

    cdef cppclass DenseExampleWiseRefinementSearchImpl(AbstractRefinementSearch):

        # Constructors:

        DenseExampleWiseRefinementSearchImpl(shared_ptr[AbstractExampleWiseRuleEvaluation] ruleEvaluationPtr,
                                             shared_ptr[Lapack] lapackPtr, uint32 numPredictions,
                                             const uint32* labelIndices, uint32 numLabels, const float64* gradients,
                                             const float64* totalSumsOfGradients, const float64* hessians,
                                             const float64* totalSumsOfHessians) except +


    cdef cppclass AbstractExampleWiseStatistics(AbstractGradientStatistics):

        # Functions:

        void setRuleEvaluation(shared_ptr[AbstractExampleWiseRuleEvaluation] ruleEvaluationPtr)


    cdef cppclass DenseExampleWiseStatisticsImpl(AbstractExampleWiseStatistics):

        # Constructors:

        DenseExampleWiseStatisticsImpl(shared_ptr[AbstractExampleWiseLoss] lossFunctionPtr,
                                       shared_ptr[AbstractExampleWiseRuleEvaluation] ruleEvaluationPtr,
                                       shared_ptr[Lapack] lapackPtr) except +


    cdef cppclass AbstractExampleWiseStatisticsFactory:

        # Functions:

        AbstractExampleWiseStatistics* create()


    cdef cppclass DenseExampleWiseStatisticsFactoryImpl(AbstractExampleWiseStatisticsFactory):

        # Constructors:

        DenseExampleWiseStatisticsFactoryImpl(shared_ptr[AbstractExampleWiseLoss] lossFunctionPtr,
                                              shared_ptr[AbstractExampleWiseRuleEvaluation] ruleEvaluationPtr,
                                              shared_ptr[Lapack] lapackPtr,
                                              shared_ptr[AbstractRandomAccessLabelMatrix] labelMatrixPtr) except +


cdef class ExampleWiseStatisticsFactory:

    # Attributes:

    cdef shared_ptr[AbstractExampleWiseStatisticsFactory] statistics_factory_ptr

    # Functions:

    cdef AbstractExampleWiseStatistics* create(self)


cdef class DenseExampleWiseStatisticsFactory(ExampleWiseStatisticsFactory):

    # Functions:

    cdef AbstractExampleWiseStatistics* create(self)


cdef class ExampleWiseStatisticsProvider(StatisticsProvider):

    # Attributes:

    cdef shared_ptr[AbstractExampleWiseStatistics] statistics_ptr

    cdef ExampleWiseRuleEvaluation rule_evaluation

    # Functions:

    cdef AbstractStatistics* get(self)


cdef class ExampleWiseStatisticsProviderFactory(StatisticsProviderFactory):

    # Attributes:

    cdef ExampleWiseLoss loss_function

    cdef ExampleWiseRuleEvaluation default_rule_evaluation

    cdef ExampleWiseRuleEvaluation rule_evaluation

    # Functions:

    cdef StatisticsProvider create(self, LabelMatrix label_matrix)
