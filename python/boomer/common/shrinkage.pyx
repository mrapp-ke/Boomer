"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for shrinking the weights of rules.
"""
from boomer.common._arrays cimport intp


cdef class Shrinkage:
    """
    A base class for all classes that implement a strategy for shrinking the weights of rules. The shrinkage parameter,
    a.k.a. the learning rate, may e.g. be constant or a function depending on the number of rules learned so far.
    """

    cdef void apply_shrinkage(self, float64[::1] predicted_scores):
        """
        Applies the shrinkage parameter to the scores that are predicted by a rule.
        
        :param predicted_scores: An array of dtype float, shape `(num_predicted_labels)`, representing the scores that 
                                 are predicted by the rule
        """
        pass


cdef class ConstantShrinkage(Shrinkage):
    """
    Shrinks the weights of rules by a constant factor.
    """

    def __cinit__(self, float shrinkage = 1.0):
        """
        :param shrinkage: The constant factor to shrink the weights of rules. Must be in (0, 1]
        """
        self.shrinkage = shrinkage

    cdef void apply_shrinkage(self, float64[::1] predicted_scores):
        cdef float shrinkage = self.shrinkage
        cdef intp num_labels = predicted_scores.shape[0]
        cdef intp c

        for c in range(num_labels):
            predicted_scores[c] *= shrinkage
