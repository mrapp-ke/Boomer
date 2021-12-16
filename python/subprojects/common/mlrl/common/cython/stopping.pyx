"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from libcpp.utility cimport move
from libcpp.memory cimport make_unique


cdef class StoppingCriterion:
    """
    A wrapper for the pure virtual C++ class `IStoppingCriterion`.
    """
    pass


cdef class SizeStoppingCriterion(StoppingCriterion):
    """
    A wrapper for the C++ class `SizeStoppingCriterion`.
    """

    def __cinit__(self, uint32 max_rules):
        """
        :param max_rules: The maximum number of rules
        """
        self.stopping_criterion_ptr = <unique_ptr[IStoppingCriterion]>make_unique[SizeStoppingCriterionImpl](max_rules)


cdef class TimeStoppingCriterion(StoppingCriterion):
    """
    A wrapper for the C++ class `TimeStoppingCriterion`.
    """

    def __cinit__(self, uint32 time_limit):
        """
        :param time_limit: The time limit in seconds
        """
        self.stopping_criterion_ptr = <unique_ptr[IStoppingCriterion]>make_unique[TimeStoppingCriterionImpl](time_limit)


cdef class AggregationFunction:
    """
    A wrapper for the pure virtual C++ class `IAggregationFunction`.
    """
    pass


cdef class MinFunction(AggregationFunction):
    """
    A wrapper for the C++ class `MinFunction`.
    """

    def __cinit__(self):
        self.aggregation_function_ptr = <unique_ptr[IAggregationFunction]>make_unique[MinFunctionImpl]()


cdef class MaxFunction(AggregationFunction):
    """
    A wrapper for the C++ class `MaxFunction`.
    """

    def __cinit__(self):
        self.aggregation_function_ptr = <unique_ptr[IAggregationFunction]>make_unique[MaxFunctionImpl]()


cdef class ArithmeticMeanFunction(AggregationFunction):
    """
    A wrapper for the C++ class `ArithmeticMeanFunction`.
    """

    def __cinit__(self):
        self.aggregation_function_ptr = <unique_ptr[IAggregationFunction]>make_unique[ArithmeticMeanFunctionImpl]()


cdef class MeasureStoppingCriterion(StoppingCriterion):
    """
    A wrapper for the C++ class `MeasureStoppingCriterion`.
    """

    def __cinit__(self, AggregationFunction aggregation_function not None, uint32 min_rules, uint32 update_interval,
                  uint32 stop_interval, uint32 num_past, uint32 num_recent, float64 min_improvement, bint force_stop):
        """
        :param aggregation_function:    The aggregation function that should be used to aggregate the scores in the
                                        buffer
        :param min_rules:               The minimum number of rules that must have been learned until the induction of
                                        rules might be stopped. Must be at least 1
        :param update_interval:         The interval to be used to update the quality of the current model, e.g., a
                                        value of 5 means that the model quality is assessed every 5 rules. Must be at
                                        least 1
        :param stop_interval:           The interval to be used to decide whether the induction of rules should be
                                        stopped, e.g., a value of 10 means that the rule induction might be stopped
                                        after 10, 20, ... rules. Must be a multiple of `updateInterval`
        :param num_past:                The number of quality scores of past iterations to be stored in a buffer. Must
                                        be at least 1
        :param num_recent:              The number of quality scores of the most recent iterations to be stored in a
                                        buffer. Must be at least 1
        :param min_improvement:         The minimum improvement in percent that must be reached for the rule induction
                                        to be continued. Must be in [0, 1]
        :param force_stop:              True, if the induction of rules should be forced to be stopped, if the stopping
                                        criterion is met, False, if the time of stopping should only be stored
        """
        self.stopping_criterion_ptr = <unique_ptr[IStoppingCriterion]>make_unique[MeasureStoppingCriterionImpl](
            move(aggregation_function.aggregation_function_ptr), min_rules, update_interval, stop_interval, num_past,
            num_recent, min_improvement, force_stop)
