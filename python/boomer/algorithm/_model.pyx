# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides model classes that are used to build rule-based models.
"""
import numpy as np


cdef class Body:
    """
    A base class for the body of a rule.
    """

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    cdef bint covers(self, float32[:] example):
        """
        Returns whether a certain example is covered by the body, or not.

        :param example: An array of dtype float, shape `(num_features)`, representing the features of an example
        :return:        1, if the example is covered, 0 otherwise
        """
        pass


cdef class EmptyBody(Body):
    """
    An empty body that matches all examples.
    """

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    cdef bint covers(self, float32[:] example):
        return 1


cdef class ConjunctiveBody(Body):
    """
    A body that consists of a conjunction of numerical conditions using <= and > operators.
    """

    def __cinit__(self, intp[::1] leq_feature_indices = None, float32[::1] leq_thresholds = None,
                  intp[::1] gr_feature_indices = None, float32[::1] gr_thresholds = None):
        """
        :param leq_feature_indices: An array of dtype int, shape `(num_leq_conditions)`, representing the features of
                                    the conditions that use the <= operator
        :param leq_thresholds:      An array of dtype float, shape `(num_leq_condition)`, representing the thresholds of
                                    the conditions that use the <= operator
        :param gr_feature_indices:  An array of dtype int, shape `(num_gr_conditions)`, representing the features of the
                                    conditions that use the > operator
        :param gr_thresholds:       An array of dtype float, shape `(num_gr_conditions)`, representing the thresholds of
                                    the conditions that use the > operator
        """
        self.leq_feature_indices = leq_feature_indices
        self.leq_thresholds = leq_thresholds
        self.gr_feature_indices = gr_feature_indices
        self.gr_thresholds = gr_thresholds

    def __getstate__(self):
        return (np.asarray(self.leq_feature_indices),
                np.asarray(self.leq_thresholds),
                np.asarray(self.gr_feature_indices),
                np.asarray(self.gr_thresholds))

    def __setstate__(self, state):
        leq_feature_indices, leq_thresholds, gr_feature_indices, gr_thresholds = state
        self.leq_feature_indices = leq_feature_indices
        self.leq_thresholds = leq_thresholds
        self.gr_feature_indices = gr_feature_indices
        self.gr_thresholds = gr_thresholds

    cdef bint covers(self, float32[:] example):
        cdef intp[::1] leq_feature_indices = self.leq_feature_indices
        cdef float32[::1] leq_thresholds = self.leq_thresholds
        cdef intp[::1] gr_feature_indices = self.gr_feature_indices
        cdef float32[::1] gr_thresholds = self.gr_thresholds
        cdef intp num_leq_conditions = leq_feature_indices.shape[0]
        cdef intp num_gr_conditions = gr_feature_indices.shape[0]
        cdef intp i, c

        for i in range(num_leq_conditions):
            c = leq_feature_indices[i]

            if example[c] > leq_thresholds[i]:
                return 0

        for i in range(num_gr_conditions):
            c = gr_feature_indices[i]

            if example[c] <= gr_thresholds[i]:
                return 0

        return 1


cdef class Head:
    """
    A base class for the head of a rule.
    """

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    cdef predict(self, float64[:] predictions):
        """
        Applies the head's prediction to a given vector of predictions.

        :param predictions: An array of dtype float, shape `(num_labels)`, representing a vector of predictions
        """
        pass


cdef class FullHead(Head):
    """
    A full head that assigns a numerical score to each label.
    """

    def __cinit__(self, float64[::1] scores = None):
        """
        :param scores:  An array of dtype float, shape `(num_labels)`, representing the scores that are predicted by the
                        rule for each label
        """
        self.scores = scores

    def __getstate__(self):
        return np.asarray(self.scores)

    def __setstate__(self, state):
        scores = state
        self.scores = scores

    cdef predict(self, float64[:] predictions):
        cdef float64[::1] scores = self.scores
        cdef intp num_cols = predictions.shape[0]
        cdef intp c

        for c in range(num_cols):
            predictions[c] += scores[c]


cdef class PartialHead(Head):
    """
    A partial head that assigns a numerical score to one or several labels.
    """

    def __cinit__(self, intp[::1] label_indices = None, float64[::1] scores = None):
        """
        :param label_indices:   An array of dtype int, shape `(num_predicted_labels)`, representing the indices of the
                                labels for which the rule predicts
        :param scores:          An array of dtype float, shape `(num_predicted_labels)`, representing the scores that
                                are predicted by the rule
        """
        self.scores = scores
        self.label_indices = label_indices

    def __getstate__(self):
        return np.asarray(self.label_indices), np.asarray(self.scores)

    def __setstate__(self, state):
        label_indices, scores = state
        self.label_indices = label_indices
        self.scores = scores

    cdef predict(self, float64[:] predictions):
        cdef intp[::1] label_indices = self.label_indices
        cdef float64[::1] scores = self.scores
        cdef intp num_labels = label_indices.shape[0]
        cdef intp c, label

        for c in range(num_labels):
            label = label_indices[c]
            predictions[label] += scores[c]


cdef class Rule:
    """
    A rule consisting of a body and head.
    """

    def __cinit__(self, body: Body = None, head: Head = None):
        """
        :param body:    The body of the rule
        :param head:    The head of the rule
        """
        self.body = body
        self.head = head

    def __getstate__(self):
        return self.body, self.head

    def __setstate__(self, state):
        body, head = state
        self.body = body
        self.head = head

    cpdef predict(self, float32[::1, :] x, float64[:, :] predictions):
        """
        Applies the rule's prediction to all examples it covers.

        :param x:               An array of dtype float, shape `(num_examples, num_features)`, representing the features
                                of the examples to predict for
        :param predictions:     An array of dtype float, shape `(num_examples, num_labels)`, representing the scores
                                predicted for the given examples
        """
        cdef Body body = self.body
        cdef Head head = self.head
        cdef intp num_examples = x.shape[0]
        cdef intp r

        for r in range(num_examples):
            if body.covers(x[r, :]):
                head.predict(predictions[r, :])
