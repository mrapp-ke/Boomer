#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement different stopping criteria that allow to decide whether additional rules should be
added to a theory or not.
"""
from abc import abstractmethod, ABC
from boomer.algorithm.model import Theory
from timeit import default_timer as timer


class StoppingCriterion(ABC):
    """
    A base class for all stopping criteria that allow to decide whether additional rules should should be added to a
    theory or not.
    """

    @abstractmethod
    def should_continue(self, theory: Theory) -> bool:
        """
        Returns, whether more rules should be added to a specific theory, or not.

        :param theory:  The theory
        :return:        True, if more rules should be added to the given theory, False otherwise
        """
        pass


class SizeStoppingCriterion(StoppingCriterion):
    """
    A stopping criterion that ensures that the number of rules in a theory does not exceed a certain maximum.
    """

    def __init__(self, num_rules: int):
        """
        :param num_rules: The maximum number of rules
        """
        self.num_rules = num_rules

    def should_continue(self, theory: Theory) -> bool:
        return len(theory) < self.num_rules


class TimeStoppingCriterion(StoppingCriterion):
    """
    A stopping criterion that ensures that a time limit is not exceeded.
    """

    def __init__(self, time_limit: int):
        """
        :param time_limit: The time limit in seconds
        """
        self.time_limit = time_limit
        self.start_time = None

    def should_continue(self, theory: Theory) -> bool:
        start_time = self.start_time

        if start_time is None:
            start_time = timer()
            self.start_time = start_time
            return True
        else:
            current_time = timer()
            run_time = current_time - start_time
            return run_time < self.time_limit
