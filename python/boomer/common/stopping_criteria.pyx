"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement different stopping criteria that allow to decide whether additional rules should be
added to a theory or not.
"""
from timeit import default_timer as timer


cdef class StoppingCriterion:
    """
    A base class for all stopping criteria that allow to decide whether additional rules should should be induced or
    not.
    """

    cdef bint should_continue(self, intp num_rules):
        """
        Returns, whether more rules should be induced, or not.

        :param num_rules:   The number of rules induced so far
        :return:            True, if more rules should be induced, False otherwise
        """
        pass


cdef class SizeStoppingCriterion(StoppingCriterion):
    """
    A stopping criterion that ensures that the number of rules does not exceed a certain maximum.
    """

    def __cinit__(self, intp max_rules):
        """
        :param max_rules: The maximum number of rules
        """
        self.max_rules = max_rules

    cdef bint should_continue(self, intp num_rules):
        cdef intp max_rules = self.max_rules
        return num_rules < max_rules


cdef class TimeStoppingCriterion(StoppingCriterion):
    """
    A stopping criterion that ensures that a time limit is not exceeded.
    """

    def __cinit__(self, intp time_limit):
        """
        :param time_limit: The time limit in seconds
        """
        self.time_limit = time_limit
        self.start_time = -1

    cdef bint should_continue(self, intp num_rules):
        cdef intp start_time = self.start_time
        cdef intp current_time, time_limit

        if start_time < 0:
            start_time = timer()
            self.start_time = start_time
            return True
        else:
            current_time = timer()
            time_limit = self.time_limit
            return current_time - start_time < time_limit
