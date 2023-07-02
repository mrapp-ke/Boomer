"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from enum import Enum

from mlrl.common.cython.validation import assert_greater_or_equal, assert_less_or_equal, assert_multiple, \
    assert_not_none


cdef class SizeStoppingCriterionConfig:
    """
    Allows to configure a stopping criterion that ensures that the number of induced rules does not exceed a certain
    maximum.
    """

    def get_max_rules(self) -> int:
        """
        Returns the maximum number of rules that are induced.

        :return: The maximum number of rules that are induced
        """
        return self.config_ptr.getMaxRules()

    def set_max_rules(self, max_rules: int) -> SizeStoppingCriterionConfig:
        """
        Sets the maximum number of rules that should be induced.

        :param max_rules:   The maximum number of rules that should be induced. Must be at least 1
        :return:            A `SizeStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        assert_greater_or_equal('max_rules', max_rules, 1)
        self.config_ptr.setMaxRules(max_rules)
        return self


cdef class TimeStoppingCriterionConfig:
    """
    Allows to configure a stopping criterion that ensures that a certain time limit is not exceeded.
    """

    def get_time_limit(self) -> int:
        """
        Returns the time limit.

        :return: The time limit in seconds
        """
        return self.config_ptr.getTimeLimit()

    def set_time_limit(self, time_limit: int) -> TimeStoppingCriterionConfig:
        """
        Sets the time limit.

        :param time_limit:  The time limit in seconds. Must be at least 1
        :return:            A `TimeStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        assert_greater_or_equal('time_limit', time_limit, 1)
        self.config_ptr.setTimeLimit(time_limit)
        return self


class AggregationFunction(Enum):
    """
    Specifies different types of aggregation functions that allow to aggregate the values that are stored in a buffer.
    """
    MIN = 0
    MAX = 1
    ARITHMETIC_MEAN = 2


cdef class PrePruningConfig:
    """
    Allow to configure a stopping criterion that stops the induction of rules as soon as the quality of a model's
    predictions for the examples in a holdout set do not improve according to a certain measure.

    This stopping criterion assesses the performance of the current model after every `updateInterval` rules and stores
    its quality in a buffer that keeps track of the last `numCurrent` iterations. If the capacity of this buffer is
    already reached, the oldest quality is passed to a buffer of size `numPast`. Every `stopInterval` rules, it is
    decided whether the rule induction should be stopped. For this reason, the `numCurrent` qualities in the first
    buffer, as well as the `numPast` qualities in the second buffer are aggregated according to a certain
    `aggregation_function`. If the percentage improvement, which results from comparing the more recent qualities from
    the first buffer to the older qualities from the second buffer, is greater than a certain `minImprovement`, the rule
    induction is continued, otherwise it is stopped.
    """

    def get_aggregation_function(self) -> AggregationFunction:
        """
        Returns the type of the aggregation function that is used to aggregate the values that are stored in a buffer.

        :return: A value of the enum `AggregationFunction` that specifies the type of the aggregation function that is
                 used to aggregate the values that are stored in a buffer
        """
        cdef uint8 enum_value = self.config_ptr.getAggregationFunction()
        return AggregationFunction(enum_value)

    def set_aggregation_function(self, aggregation_function: AggregationFunction) -> PrePruningConfig:
        """
        Sets the type of the aggregation function that should be used to aggregate the values that are stored in a
        buffer.

        :param aggregation_function:    A value of the enum `AggregationFunction` that specifies the type of the
                                        aggregation function that should be used to aggregate the values that are stored
                                        in a buffer
        :return:                        A `PrePruningConfig` that allows further configuration of the stopping criterion
        """
        assert_not_none('aggregation_function', aggregation_function)
        cdef uint8 enum_value = aggregation_function.value
        self.config_ptr.setAggregationFunction(<AggregationFunctionImpl>enum_value)
        return self

    def is_holdout_set_used(self) -> bool:
        """
        Returns whether the quality of the current model's predictions is measured on the holdout set, if available, or
        if the training set is used instead.

        :return: True, if the quality of the current model's predictions is measured on the holdout set, if available,
                 False, if the training set is used instead
        """
        return self.config_ptr.isHoldoutSetUsed()

    def set_use_holdout_set(self, use_holdout_set: bool) -> PrePruningConfig:
        """
        Sets whether the quality of he current model's predictions should be measured on the holdout set, if available,
        or if the training set should be used instead.

        :param use_holdout_set: True, if the quality of the current model's predictions should be measured on the
                                holdout set, if available, False, if the training set should be used instead
        :return:                A `PrePruningConfig` that allows further configuration of the stopping criterion
        """
        self.config_ptr.setUseHoldoutSet(use_holdout_set)
        return self

    def is_remove_unused_rules(self) -> bool:
        """
        Returns whether rules that have been induced, but are not used, should be removed from the final model or not.

        :return: True, if unused rules should be removed from the model, False otherwise
        """
        return self.config_ptr.isRemoveUnusedRules()

    def set_remove_unused_rules(self, remove_unused_rules: bool) -> PrePruningConfig:
        """
        Sets whether rules that have been induced, but are not used, should be removed from the final model or not.

        :param remove_unused_rules: True, if unused rules should be removed from the model, false otherwise
        :return:                    A `PrePruningConfig` that allows further configuration of the stopping criterion
        """
        self.config_ptr.setRemoveUnusedRules(remove_unused_rules)
        return self

    def get_min_rules(self) -> int:
        """
        Returns the minimum number of rules that must have been learned until the induction of rules might be stopped.

        :return: The minimum number of rules that must have been learned until the induction of rules might be stopped
        """
        return self.config_ptr.getMinRules()

    def set_min_rules(self, min_rules: int) -> PrePruningConfig:
        """
        Sets the minimum number of rules that must have been learned until the induction of rules might be stopped.

        :param min_rules:   The minimum number of rules that must have been learned until the induction of rules might
                            be stopped. Must be at least 1
        :return:            A `PrePruningConfig` that allows further configuration of the stopping criterion
        """
        assert_greater_or_equal('min_rules', min_rules, 1)
        self.config_ptr.setMinRules(min_rules)
        return self

    def get_update_interval(self) -> int:
        """
        Returns the interval that is used to update the quality of the current model.

        :return: The interval that is used to update the quality of the current model
        """
        return self.config_ptr.getUpdateInterval()

    def set_update_interval(self, update_interval: int) -> PrePruningConfig:
        """
        Sets the interval that should be used to update the quality of the current model.

        :param update_interval: The interval that should be used to update the quality of the current model, e.g., a
         *                      value of 5 means that the model quality is assessed every 5 rules. Must be at least 1
        :return:                A `PrePruningConfig` that allows further configuration of the stopping criterion
        """
        assert_greater_or_equal('update_interval', update_interval, 1)
        self.config_ptr.setUpdateInterval(update_interval)
        return self

    def get_stop_interval(self) -> int:
        """
        Returns the interval that is used to decide whether the induction of rules should be stopped.

        :return: The interval that is used to decide whether the induction of rules should be stopped
        """
        return self.config_ptr.getStopInterval()

    def set_stop_interval(self, stop_interval: int) -> PrePruningConfig:
        """
        Sets the interval that should be used to decide whether the induction of rules should be stopped.

        :param stop_interval:   The interval that should be used to decide whether the induction of rules should be
                                stopped, e.g., a value of 10 means that the rule induction might be stopped after 10,
                                20, ... rules. Must be a multiple of the update interval
        :return:                A `PrePruningConfig` that allows further configuration of the stopping criterion
        """
        assert_multiple('stop_interval', stop_interval, self.config_ptr.getUpdateInterval())
        self.config_ptr.setStopInterval(stop_interval)
        return self

    def get_num_past(self) -> int:
        """
        Returns the number of quality stores of past iterations that are stored in a buffer.

        :return: The number of quality stores of past iterations that are stored in a buffer
        """
        return self.config_ptr.getNumPast()

    def set_num_past(self, num_past: int) -> PrePruningConfig:
        """
        Sets the number of past iterations that should be stored in a buffer.

        :param num_past:    The number of past iterations that should be be stored in a buffer. Must be at least 1
        :return:            A `PrePruningConfig` that allows further configuration of the stopping criterion
        """
        assert_greater_or_equal('num_past', num_past, 1)
        self.config_ptr.setNumPast(num_past)
        return self

    def get_num_current(self) -> int:
        """
        Returns the number of the most recent iterations that are stored in a buffer.

        :return: The number of the most recent iterations that are stored in a buffer
        """
        return self.config_ptr.getNumCurrent()

    def set_num_current(self, num_current: int) -> PrePruningConfig:
        """
        Sets the number of the most recent iterations that should be stored in a buffer.

        :param num_current: The number of the most recent iterations that should be stored in a buffer. Must be at least
                            1
        :return:            A `PrePruningConfig` that allows further configuration of the stopping criterion
        """
        assert_greater_or_equal('num_current', num_current, 1)
        self.config_ptr.setNumCurrent(num_current)
        return self

    def get_min_improvement(self) -> float:
        """
        Returns the minimum improvement that must be reached for the rule induction to be continued.

        :return: The minimum improvement that must be reached for the rule induction to be continued
        """
        return self.config_ptr.getMinImprovement()

    def set_min_improvement(self, min_improvement: float) -> PrePruningConfig:
        """
        Sets the minimum improvement that must be reached for the rule induction to be continued.

        :param min_improvement: The minimum improvement in percent that must be reached for the rule induction to be
                                continued. Must be in [0, 1]
        :return:                A `PrePruningConfig` that allows further configuration of the stopping criterion
        """
        assert_greater_or_equal('min_improvement', min_improvement, 0)
        assert_less_or_equal('min_improvement', min_improvement, 1)
        self.config_ptr.setMinImprovement(min_improvement)
        return self


cdef class PostPruningConfig:
    """
    Defines an interface for all classes that allow to configure a stopping criterion that keeps track of the number of
    rules in a model that perform best with respect to the examples in the training or holdout set according to a
    certain measure.

    This stopping criterion assesses the performance of the current model after every `interval` rules and stores and
    checks whether the current model is the best one evaluated so far.
    """

    def is_holdout_set_used(self) -> bool:
        """
        Returns whether the quality of the current model's predictions is measured on the holdout set, if available, or
        if the training set is used instead.

        :return: True, if the quality of the current model's predictions is measured on the holdout set, if available,
                 False, if the training set is used instead
        """
        return self.config_ptr.isHoldoutSetUsed()

    def set_use_holdout_set(self, use_holdout_set: bool) -> PostPruningConfig:
        """
        Sets whether the quality of he current model's predictions should be measured on the holdout set, if available,
        or if the training set should be used instead.

        :param use_holdout_set: True, if the quality of the current model's predictions should be measured on the
                                holdout set, if available, False, if the training set should be used instead
        :return:                A `PostPruningConfig` that allows further configuration of the stopping criterion
        """
        self.config_ptr.setUseHoldoutSet(use_holdout_set)
        return self

    def is_remove_unused_rules(self) -> bool:
        """
        Returns whether rules that have been induced, but are not used, should be removed from the final model or not.

        :return: True, if unused rules should be removed from the model, False otherwise
        """
        return self.config_ptr.isRemoveUnusedRules()

    def set_remove_unused_rules(self, remove_unused_rules: bool) -> PostPruningConfig:
        """
        Sets whether rules that have been induced, but are not used, should be removed from the final model or not.

        :param remove_unused_rules: True, if unused rules should be removed from the model, false otherwise
        :return:                    A `PostPruningConfig` that allows further configuration of the stopping criterion
        """
        self.config_ptr.setRemoveUnusedRules(remove_unused_rules)
        return self

    def get_min_rules(self) -> int:
        """
        Returns the minimum number of rules that must be included in a model.

        :return: The minimum number of rules that must be included in a model
        """
        return self.config_ptr.getMinRules()

    def set_min_rules(self, min_rules: int) -> PostPruningConfig:
        """
        Sets the minimum number of rules that must be included in a model.

        :param min_rules:   The minimum number of rules that must be included in a model. Must be at least 1
        :return:            A `PostPruningConfig` that allows further configuration of the stopping criterion
        """
        assert_greater_or_equal('min_rules', min_rules, 1)
        self.config_ptr.setMinRules(min_rules)
        return self

    def get_interval(self) -> int:
        """
        Returns the interval that is used to check whether the current model is the best one evaluated so far.

        :return: The interval that is used to check whether the current model is the best one evaluated so far
        """
        return self.config_ptr.getInterval()

    def set_interval(self, interval: int) -> PostPruningConfig:
        """
        Sets the interval that should be used to check whether the current model is the best one evaluated so far.

        :param interval:    The interval that should be used to check whether the current model is the best one
                            evaluated so far, e.g., a value of 10 means that the best model may include 10, 20, ...
                            rules
        :return:            A `PostPruningConfig` that allows further configuration of the stopping criterion
        """
        assert_greater_or_equal('interval', interval, 1)
        self.config_ptr.setInterval(interval)
        return self
