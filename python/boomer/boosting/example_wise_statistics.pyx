"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store gradients and Hessians that are calculated according to a
(non-decomposable) loss function that is applied example-wise.
"""
from boomer.common.input_data cimport RandomAccessLabelMatrix, AbstractLabelMatrix
from boomer.boosting._lapack cimport init_lapack
from boomer.boosting.example_wise_losses cimport ExampleWiseLoss
from boomer.boosting.example_wise_rule_evaluation cimport ExampleWiseRuleEvaluation

from libcpp.memory cimport make_shared, dynamic_pointer_cast


cdef class ExampleWiseStatisticsFactory:
    """
    A wrapper for the abstract C++ class `AbstractExampleWiseStatisticsFactory`.
    """

    cdef AbstractExampleWiseStatistics* create(self):
        """
        Creates a new instance of the class `AbstractExampleWiseStatistics`.

        :return: A pointer to an object of type `AbstractExampleWiseStatistics` that has been created
        """
        return self.statistics_factory_ptr.get().create()


cdef class DenseExampleWiseStatisticsFactory(ExampleWiseStatisticsFactory):
    """
    A wrapper for the C++ class `DenseExampleWiseStatisticsFactoryImpl`.
    """

    def __cinit__(self, ExampleWiseLoss loss_function, ExampleWiseRuleEvaluation rule_evaluation,
                 RandomAccessLabelMatrix label_matrix):
        """
        :param loss_function:   The loss function to be used for calculating gradients and Hessians
        :param rule_evaluation: The `LabelWiseRuleEvaluation` to be used for calculating the predictions, as well as
                                corresponding quality scores, of rules
        :param label_matrix:    A `RandomAccessLabelMatrix` that provides random access to the labels of the training
                                examples
        """
        self.statistics_factory_ptr = <shared_ptr[AbstractExampleWiseStatisticsFactory]>make_shared[DenseExampleWiseStatisticsFactoryImpl](
            loss_function.loss_function_ptr, rule_evaluation.rule_evaluation_ptr, shared_ptr[Lapack](init_lapack()),
            dynamic_pointer_cast[AbstractRandomAccessLabelMatrix, AbstractLabelMatrix](label_matrix.label_matrix_ptr))


cdef class ExampleWiseStatisticsProvider(StatisticsProvider):
    """
    Provides access to an object of type `AbstractExampleWiseStatistics`.
    """

    def __cinit__(self, ExampleWiseStatisticsFactory statistics_factory, ExampleWiseRuleEvaluation rule_evaluation):
        """
        :param statistics_factory:  A factory that allows to create a new object of type `AbstractExampleWiseStatistics`
        :param rule_evaluation:     The `ExampleWiseRuleEvaluation` to switch to when invoking the function
                                    `switch_rule_evaluation`
        """
        self.statistics_ptr = shared_ptr[AbstractExampleWiseStatistics](statistics_factory.create())
        self.rule_evaluation = rule_evaluation

    cdef AbstractStatistics* get(self):
        return self.statistics_ptr.get()

    cdef void switch_rule_evaluation(self):
        cdef ExampleWiseRuleEvaluation rule_evaluation = self.rule_evaluation
        cdef shared_ptr[AbstractExampleWiseRuleEvaluation] rule_evaluation_ptr = rule_evaluation.rule_evaluation_ptr
        self.statistics_ptr.get().setRuleEvaluation(rule_evaluation_ptr)


cdef class ExampleWiseStatisticsProviderFactory(StatisticsProviderFactory):
    """
    A factory that allows to create instances of the class `ExampleWiseStatisticsProvider`.
    """

    def __cinit__(self, ExampleWiseLoss loss_function, ExampleWiseRuleEvaluation default_rule_evaluation,
                  ExampleWiseRuleEvaluation rule_evaluation):
        """
        :param loss_function:           The loss function to be used for calculating gradients and Hessians
        :param default_rule_evaluation: The `ExampleWiseRuleEvaluation` to be used for calculating the predictions, as
                                        well as corresponding quality scores, of the default rules
        :param rule_evaluation:         The `ExampleWiseRuleEvaluation` to be used for calculating the predictions, as
                                        well as corresponding quality scores, of rules
        :param label_matrix:            A label matrix that provides random access to the labels of the training
                                        examples
        """
        self.loss_function = loss_function
        self.default_rule_evaluation = default_rule_evaluation
        self.rule_evaluation = rule_evaluation

    cdef ExampleWiseStatisticsProvider create(self, LabelMatrix label_matrix):
        cdef ExampleWiseStatisticsFactory statistics_factory

        if isinstance(label_matrix, RandomAccessLabelMatrix):
            statistics_factory = DenseExampleWiseStatisticsFactory.__new__(DenseExampleWiseStatisticsFactory,
                                                                           self.loss_function,
                                                                           self.default_rule_evaluation, label_matrix)
        else:
            raise ValueError('Unsupported type of label matrix: ' + str(label_matrix.__type__))

        return ExampleWiseStatisticsProvider.__new__(ExampleWiseStatisticsProvider, statistics_factory,
                                                     self.rule_evaluation)
