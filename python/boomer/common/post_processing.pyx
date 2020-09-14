"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that allow to post-process the predictions of rules once they have been learned.
"""


cdef class PostProcessor:
    """
    A base class for all classes that allow to post-process the predictions rules once they have been learned.
    """

    cdef void post_process(self, Prediction* prediction):
        """
        Post-processes the predictions of a rule.

        :param prediction: A pointer to an object of type `Prediction`, representing the predictions of the rule
        """
        pass
