"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides wrappers for classes that allow to store statistics about the labels of training examples.
"""


cdef class StatisticsProvider:
    """
    Provides access to an object of type `AbstractStatistics`.
    """

    cdef AbstractStatistics* get(self):
        """
        Returns a pointer to an object of type `AbstractStatistics`.

        :return: A pointer to an object of type `AbstractStatistics`
        """
        pass

    cdef void switch_rule_evaluation(self):
        """
        Allows to switch the implementation that is used for calculating the predictions of rules, as well as
        corresponding quality scores, from the one that was iniatally used for the default rule to another that will be
        used for all remaining rules.
        """
        pass


cdef class StatisticsProviderFactory:
    """
    A factory that allows to create instances of the class `StatisticsProvider`.
    """

    cdef StatisticsProvider create(self, LabelMatrix label_matrix):
        """
        Creates and returns a new instance of the class `StatisticsProvider`.

        :param label_matrix:    A `LabelMatrix` that provides access to the labels of the training examples
        :return:                The `StatisticsProvider` that has been created
        """
        pass
