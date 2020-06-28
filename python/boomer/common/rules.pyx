"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides model classes that are used to build rule-based models.
"""
from boomer.common._arrays cimport array_uint32, array_intp, array_float32, c_matrix_uint8, c_matrix_float64

from cython.operator cimport dereference, postincrement

import numpy as np


cdef class Body:
    """
    A base class for the body of a rule.
    """

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    cdef bint covers(self, float32[::1] example):
        """
        Returns whether a certain example is covered by the body, or not.

        The feature values of the example must be given as a dense C-contiguous array.

        :param example: An array of dtype float, shape `(num_features)`, representing the features of an example
        :return:        1, if the example is covered, 0 otherwise
        """
        pass

    cdef bint covers_sparse(self, float32[::1] example_data, intp[::1] example_indices, float32[::1] tmp_array1,
                            uint32[::1] tmp_array2, uint32 n):
        """
        Returns whether a certain example is covered by the body, or not.

        The feature values of the example must be given as a sparse array.

        :param example_data:    An array of dtype float, shape `(num_non_zero_feature_values), representing the non-zero
                                feature values of the training examples
        :param example_indices: An array of dtype int, shape `(num_non_zero_feature_values)`, representing the indices
                                of the features, the values in `example_data` correspond to
        :param tmp_array1:      An array of dtype float, shape `(num_features)` that is used to temporarily store
                                non-zero feature values. May contain arbitrary values
        :param tmp_array2:      An array of dtype uint, shape `(num_features)` that is used to temporarily keep track of
                                the feature indices with non-zero feature values. Must not contain any elements with
                                value `n`
        :param n:               An arbitrary number. If this function is called multiple times for different examples,
                                but using the same `tmp_array2`, the number must be unique for each of the function
                                invocations
        :return:                1, if the example is covered, 0 otherwise
        """


cdef class EmptyBody(Body):
    """
    An empty body that matches all examples.
    """

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    cdef bint covers(self, float32[::1] example):
        return True

    cdef bint covers_sparse(self, float32[::1] example_data, intp[::1] example_indices, float32[::1] tmp_array1,
                            uint32[::1] tmp_array2, uint32 n):
        return True


cdef class ConjunctiveBody(Body):
    """
    A body that consists of a conjunction of conditions using the operators <= or > for numerical conditions and = or !=
    for nominal conditions, respectively.
    """

    def __cinit__(self, intp[::1] leq_feature_indices = None, float32[::1] leq_thresholds = None,
                  intp[::1] gr_feature_indices = None, float32[::1] gr_thresholds = None,
                  intp[::1] eq_feature_indices = None, float32[::1] eq_thresholds = None,
                  intp[::1] neq_feature_indices = None, float32[::1] neq_thresholds = None):
        """
        :param leq_feature_indices: An array of dtype int, shape `(num_leq_conditions)`, representing the indices of the
                                    features, the numerical conditions that use the <= operator correspond to or None,
                                    if the body does not contain such a condition
        :param leq_thresholds:      An array of dtype float, shape `(num_leq_condition)`, representing the thresholds of
                                    the numerical conditions that use the <= operator or None, if the body does not
                                    contain such a condition
        :param gr_feature_indices:  An array of dtype int, shape `(num_gr_conditions)`, representing the indices of the
                                    features, the numerical conditions that use the > operator correspond to or None, if
                                    the body does not contain such a condition
        :param gr_thresholds:       An array of dtype float, shape `(num_gr_conditions)`, representing the thresholds of
                                    the numerical conditions that use the > operator or None, if the body does not
                                    contain such a condition
        :param eq_feature_indices:  An array of dtype int, shape `(num_eq_conditions)`, representing the indices of the
                                    features, the nominal conditions that use the = operator correspond to or None, if
                                    the body does not contain such a condition
        :param eq_thresholds:       An array of dtype float, shape `(num_eq_conditions)`, representing the thresholds of
                                    the nominal conditions that use the = operator or None, if the body does not contain
                                    such a condition
        :param neq_feature_indices: An array of dtype int, shape `(num_neq_conditions)`, representing the indices of the
                                    features, the nominal conditions that use the != operator correspond to or None, if
                                    the body does not contain such a condition
        :param neq_thresholds:      An array of dtype float, shape `(num_neq_conditions)`, representing the thresholds
                                    of the nominal conditions that use the != operator or None, if the body does not
                                    contain such a condition
        """
        self.leq_feature_indices = leq_feature_indices
        self.leq_thresholds = leq_thresholds
        self.gr_feature_indices = gr_feature_indices
        self.gr_thresholds = gr_thresholds
        self.eq_feature_indices = eq_feature_indices
        self.eq_thresholds = eq_thresholds
        self.neq_feature_indices = neq_feature_indices
        self.neq_thresholds = neq_thresholds

    def __getstate__(self):
        return (np.asarray(self.leq_feature_indices) if self.leq_feature_indices is not None else None,
                np.asarray(self.leq_thresholds) if self.leq_thresholds is not None else None,
                np.asarray(self.gr_feature_indices) if self.gr_feature_indices is not None else None,
                np.asarray(self.gr_thresholds) if self.gr_thresholds is not None else None,
                np.asarray(self.eq_feature_indices) if self.eq_feature_indices is not None else None,
                np.asarray(self.eq_thresholds) if self.eq_thresholds is not None else None,
                np.asarray(self.neq_feature_indices) if self.neq_feature_indices is not None else None,
                np.asarray(self.neq_thresholds) if self.neq_thresholds is not None else None)

    def __setstate__(self, state):
        self.leq_feature_indices = state[0]
        self.leq_thresholds = state[1]
        self.gr_feature_indices = state[2]
        self.gr_thresholds = state[3]
        self.eq_feature_indices = state[4]
        self.eq_thresholds = state[5]
        self.neq_feature_indices = state[6]
        self.neq_thresholds = state[7]

    cdef bint covers(self, float32[::1] example):
        cdef intp[::1] feature_indices = self.leq_feature_indices
        cdef float32[::1] thresholds = self.leq_thresholds
        cdef intp num_conditions = feature_indices.shape[0]
        cdef intp i, c

        for i in range(num_conditions):
            c = feature_indices[i]

            if example[c] > thresholds[i]:
                return False

        feature_indices = self.gr_feature_indices
        thresholds = self.gr_thresholds
        num_conditions = feature_indices.shape[0]

        for i in range(num_conditions):
            c = feature_indices[i]

            if example[c] <= thresholds[i]:
                return False

        feature_indices = self.eq_feature_indices
        thresholds = self.eq_thresholds
        num_conditions = feature_indices.shape[0]

        for i in range(num_conditions):
            c = feature_indices[i]

            if example[c] != thresholds[i]:
                return False

        feature_indices = self.neq_feature_indices
        thresholds = self.neq_thresholds
        num_conditions = feature_indices.shape[0]

        for i in range(num_conditions):
            c = feature_indices[i]

            if example[c] == thresholds[i]:
                return False

        return True

    cdef bint covers_sparse(self, float32[::1] example_data, intp[::1] example_indices, float32[::1] tmp_array1,
                            uint32[::1] tmp_array2, uint32 n):
        cdef intp num_non_zero_feature_values = example_data.shape[0]
        cdef intp i, c

        for i in range(num_non_zero_feature_values):
            c = example_indices[i]
            tmp_array1[c] = example_data[i]
            tmp_array2[c] = n

        cdef intp[::1] feature_indices = self.leq_feature_indices
        cdef float32[::1] thresholds = self.leq_thresholds
        cdef intp num_conditions = feature_indices.shape[0]
        cdef float32 feature_value

        for i in range(num_conditions):
            c = feature_indices[i]
            feature_value = tmp_array1[c] if tmp_array2[c] == n else 0

            if feature_value > thresholds[i]:
                return False

        feature_indices = self.gr_feature_indices
        thresholds = self.gr_thresholds
        num_conditions = feature_indices.shape[0]

        for i in range(num_conditions):
            c = feature_indices[i]
            feature_value = tmp_array1[c] if tmp_array2[c] == n else 0

            if feature_value <= thresholds[i]:
                return False

        feature_indices = self.eq_feature_indices
        thresholds = self.eq_thresholds
        num_conditions = feature_indices.shape[0]

        for i in range(num_conditions):
            c = feature_indices[i]
            feature_value = tmp_array1[c] if tmp_array2[c] == n else 0

            if feature_value != thresholds[i]:
                return False

        feature_indices = self.neq_feature_indices
        thresholds = self.neq_thresholds
        num_conditions = feature_indices.shape[0]

        for i in range(num_conditions):
            c = feature_indices[i]
            feature_value = tmp_array1[c] if tmp_array2[c] == n else 0

            if feature_value == thresholds[i]:
                return False

        return True


cdef class Head:
    """
    A base class for the head of a rule.
    """

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    cdef void predict(self, float64[::1] predictions, uint8[::1] mask = None):
        """
        Applies the head's prediction to a given vector of predictions. Optionally, the prediction can be restricted to
        certain labels.

        :param mask:        An array of dtype uint, shape `(num_labels)`, indicating for which labels it is allowed to
                            predict or None, if the prediction should not be restricted
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

    cdef void predict(self, float64[::1] predictions, uint8[::1] mask = None):
        cdef float64[::1] scores = self.scores
        cdef intp num_cols = predictions.shape[0]
        cdef intp c

        for c in range(num_cols):
            if mask is not None:
                if mask[c]:
                    predictions[c] += scores[c]
                    mask[c] = False
            else:
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

    cdef void predict(self, float64[::1] predictions, uint8[::1] mask = None):
        cdef intp[::1] label_indices = self.label_indices
        cdef float64[::1] scores = self.scores
        cdef intp num_labels = label_indices.shape[0]
        cdef intp c, l

        for c in range(num_labels):
            l = label_indices[c]

            if mask is not None:
                if mask[l]:
                    predictions[l] += scores[c]
                    mask[l] = False
            else:
                predictions[l] += scores[c]


cdef class Rule:
    """
    A rule consisting of a body and head.
    """

    def __cinit__(self, Body body = None, Head head= None):
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

    cdef predict(self, float32[:, ::1] x, float64[:, ::1] predictions, uint8[:, ::1] mask = None):
        """
        Applies the rule's prediction to a matrix of predictions for all examples it covers. Optionally, the prediction
        can be restricted to certain examples and labels.

        The feature matrix must be given as a dense C-contiguous array.

        :param x:               An array of dtype float, shape `(num_examples, num_features)`, representing the features
                                of the examples to predict for
        :param predictions:     An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                predictions for individual examples and labels
        :param mask:            An array of dtype uint, shape `(num_examples, num_labels)`, indicating for which
                                examples and labels it is allowed to predict or None, if the prediction should not be
                                restricted
        """
        cdef Body body = self.body
        cdef Head head = self.head
        cdef intp num_examples = x.shape[0]
        cdef uint8[::1] mask_row
        cdef intp r

        for r in range(num_examples):
            if body.covers(x[r, :]):
                mask_row = None if mask is None else mask[r, :]
                head.predict(predictions[r, :], mask_row)

    cdef predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices, intp num_features,
                     float32[::1] tmp_array1, uint32[::1] tmp_array2, uint32 n, float64[:, ::1] predictions,
                     uint8[:, ::1] mask = None):
        """
        Applies the rule's predictions to a matrix of predictions for all examples it covers. Optionally, the prediction
        can be restricted to certain examples and labels.

        The feature matrix must be given in compressed sparse row (CSR) format.

        :param x_data:          An array of dtype float, shape `(num_non_zero_feature_values)`, representing the
                                non-zero feature values of the examples to predict for
        :param x_row_indices:   An array of dtype int, shape `(num_examples + 1)`, representing the indices of the first
                                element in `x_data` and `x_col_indices` that corresponds to a certain examples. The
                                index at the last position is equal to `num_non_zero_feature_values`
        :param x_col_indices:   An array of dtype int, shape `(num_non_zero_feature_values)`, representing the
                                column-indices of the examples, the values in `x_data` correspond to
        :param num_features:    The total number of features
        :param tmp_array1:      An array of dtype float, shape `(num_features)` that is used to temporarily store
                                non-zero feature values. May contain arbitrary values
        :param tmp_array2:      An array of dtype uint, shape `(num_features)` that is used to temporarily keep track of
                                the feature indices with non-zero feature values. Must not contain any elements with
                                value `n`
        :param n:               An arbitrary number. If this function is called multiple times on different rules, but
                                using the same `tmp_array2`, the number must be unique for each of the function
                                invocations and the numbers `n...n + num_examples` must not be used for any of the
                                remaining invocations
        :param predictions:     An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                predictions of individual examples and labels
        :param mask:            An array of dtype uint, shape `(num_examples, num_labels)`, indicating for which
                                examples and labels it is allowed to predict or None, if the prediction should not be
                                restricted
        """
        cdef Body body = self.body
        cdef Head head = self.head
        cdef intp num_examples = x_row_indices.shape[0] - 1
        cdef uint32 current_n = n
        cdef uint8[::1] mask_row
        cdef intp r, start, end

        for r in range(num_examples):
            start = x_row_indices[r]
            end = x_row_indices[r + 1]

            if body.covers_sparse(x_data[start:end], x_col_indices[start:end], tmp_array1, tmp_array2, current_n):
                mask_row = None if mask is None else mask[r, :]
                head.predict(predictions[r, :], mask_row)

            current_n += 1


cdef class RuleModel:
    """
    A base class for all rule-based models.
    """

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    cdef void add_rule(self, Rule rule):
        """
        Adds a new rule to the model.
        
        :param rule: The rule to be added
        """
        pass

    cdef float64[:, ::1] predict(self, float32[:, ::1] x, intp num_labels):
        """
        Aggregates and returns the predictions provided by several rules.

        The feature matrix must be given as a dense C-contiguous array.
        
        :param x:           An array of dtype float, shape `(num_examples, num_features)`, representing the features of
                            the examples to predict for
        :param num_labels:  The total number of labels
        :return:            An array of dtype float, shape `(num_examples, num_labels)`, representing the predictions
                            for individual examples and labels
        """
        pass

    cdef float64[:, ::1] predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                                     intp num_features, intp num_labels):
        """
        Aggregates and returns the predictions provided by several rules.

        The feature matrix must be given in compressed sparse row (CSR) format.
        
        :param x_data:          An array of dtype float, shape `(num_non_zero_feature_values)`, representing the
                                non-zero feature values of the training examples 
        :param x_row_indices:   An array of dtype int, shape `(num_examples + 1)`, representing the indices of the first
                                element in `x_data` and `x_col_indices` that corresponds to a certain examples. The
                                index at the last position is equal to `num_non_zero_feature_values`
        :param x_col_indices:   An array of dtype int, shape `(num_non_zero_feature_values)`, representing the
                                column-indices of the examples, the values in `x_data` correspond to
        :param num_features:    The total number of features
        :param num_labels:      The total number of labels
        :return:                An array of dtype float, shape `(num_examples, num_labels)`, representing the
                                predictions for individual examples and labels
        """
        pass


cdef class RuleList(RuleModel):
    """
    A model that stores several rules in a list.
    """

    def __cinit__(self, bint use_mask = False):
        """
        :param use_mask: True, if only one rule is allowed to predict per label, False otherwise
        """
        self.use_mask = use_mask
        self.rules = []

    def __getstate__(self):
        return self.use_mask, self.rules

    def __setstate__(self, state):
        use_mask, rules = state
        self.use_mask = use_mask
        self.rules = rules

    cdef void add_rule(self, Rule rule):
        cdef list rules = self.rules
        rules.append(rule)

    cdef float64[:, ::1] predict(self, float32[:, ::1] x, intp num_labels):
        cdef intp num_examples = x.shape[0]
        cdef float64[:, ::1] predictions = c_matrix_float64(num_examples, num_labels)
        predictions[:, :] = 0
        cdef bint use_mask = self.use_mask
        cdef uint8[:, ::1] mask

        if use_mask:
            mask = c_matrix_uint8(num_examples, num_labels)
            mask[:, :] = True
        else:
            mask = None

        cdef list rules = self.rules
        cdef Rule rule

        for rule in rules:
            rule.predict(x, predictions, mask)

        return predictions

    cdef float64[:, ::1] predict_csr(self, float32[::1] x_data, intp[::1] x_row_indices, intp[::1] x_col_indices,
                                     intp num_features, intp num_labels):
        cdef intp num_examples = x_row_indices.shape[0] - 1
        cdef float64[:, ::1] predictions = c_matrix_float64(num_examples, num_labels)
        predictions[:, :] = 0
        cdef bint use_mask = self.use_mask
        cdef uint8[:, ::1] mask

        if use_mask:
            mask = c_matrix_uint8(num_examples, num_labels)
            mask[:, :] = True
        else:
            mask = None

        cdef float32[::1] tmp_array1 = array_float32(num_features)
        cdef uint32[::1] tmp_array2 = array_uint32(num_features)
        tmp_array2[:] = 0
        cdef uint32 n = 1
        cdef list rules = self.rules
        cdef Rule rule

        for rule in rules:
            rule.predict_csr(x_data, x_row_indices, x_col_indices, num_features, tmp_array1, tmp_array2, n, predictions,
                             mask)
            n += 1

        return predictions


cdef class ModelBuilder:
    """
    A base class for all builders that allow to incrementally build a `RuleModel`.
    """

    cdef void set_default_rule(self, float64[::1] scores):
        """
        Initializes the model and sets its default rule.

        :param scores: An array of dtype float, shape `(num_labels)`, representing the scores that are predicted by the
                       default rule for each label
        """
        pass

    cdef void add_rule(self, intp[::1] label_indices, float64[::1] scores, double_linked_list[Condition] conditions,
                       intp[::1] num_conditions_per_comparator):
        """
        Adds a new rule to the model.

        :param label_indices:                   An array of dtype int, shape `(num_predicted_labels)`, representing the
                                                indices of the labels for which the rule predicts or None, if the rule
                                                predicts for all labels
        :param scores:                          An array of dtype float, shape `(num_predicted_labels)`, representing
                                                the scores that are predicted by the rule
        :param conditions:                      A list that contains the rule's conditions
        :param num_conditions_per_comparator:   An array of dtype int, shape `(4)`, representing the number of
                                                conditions that use a specific operator
        """
        pass

    cdef RuleModel build_model(self):
        """
        Builds and returns the model.

        :return: The model that has been built
        """
        pass


cdef class RuleListBuilder(ModelBuilder):
    """
    A builder that allows to incrementally build a `RuleList`.
    """

    def __cinit__(self, bint use_mask = False, bint default_rule_at_end = False):
        self.use_mask = use_mask
        self.default_rule_at_end = default_rule_at_end
        self.rule_list = None
        self.default_rule = None

    cdef void set_default_rule(self, float64[::1] scores):
        cdef bint use_mask = self.use_mask
        cdef bint default_rule_at_end = self.default_rule_at_end
        cdef RuleList rule_list = RuleList.__new__(RuleList, use_mask)
        cdef FullHead head = FullHead.__new__(FullHead, scores)
        cdef EmptyBody body = EmptyBody.__new__(EmptyBody)
        cdef Rule default_rule = Rule.__new__(Rule, body, head)

        if default_rule_at_end:
            self.default_rule = default_rule
        else:
            rule_list.add_rule(default_rule)

        self.rule_list = rule_list

    cdef void add_rule(self, intp[::1] label_indices, float64[::1] scores, double_linked_list[Condition] conditions,
                       intp[::1] num_conditions_per_comparator):
        cdef intp num_conditions = num_conditions_per_comparator[<intp>Comparator.LEQ]
        cdef intp[::1] leq_feature_indices = array_intp(num_conditions) if num_conditions > 0 else None
        cdef float32[::1] leq_thresholds = array_float32(num_conditions) if num_conditions > 0 else None
        num_conditions = num_conditions_per_comparator[<intp>Comparator.GR]
        cdef intp[::1] gr_feature_indices = array_intp(num_conditions) if num_conditions > 0 else None
        cdef float32[::1] gr_thresholds = array_float32(num_conditions) if num_conditions > 0 else None
        num_conditions = num_conditions_per_comparator[<intp>Comparator.EQ]
        cdef intp[::1] eq_feature_indices = array_intp(num_conditions) if num_conditions > 0 else None
        cdef float32[::1] eq_thresholds = array_float32(num_conditions) if num_conditions > 0 else None
        num_conditions = num_conditions_per_comparator[<intp>Comparator.NEQ]
        cdef intp[::1] neq_feature_indices = array_intp(num_conditions) if num_conditions > 0 else None
        cdef float32[::1] neq_thresholds = array_float32(num_conditions) if num_conditions > 0 else None
        cdef double_linked_list[Condition].iterator iterator = conditions.begin()
        cdef intp leq_i = 0
        cdef intp gr_i = 0
        cdef intp eq_i = 0
        cdef intp neq_i = 0
        cdef Condition condition
        cdef Comparator comparator

        while iterator != conditions.end():
            condition = dereference(iterator)
            comparator = condition.comparator

            if comparator == Comparator.LEQ:
               leq_feature_indices[leq_i] = condition.feature_index
               leq_thresholds[leq_i] = condition.threshold
               leq_i += 1
            elif comparator == Comparator.GR:
               gr_feature_indices[gr_i] = condition.feature_index
               gr_thresholds[gr_i] = condition.threshold
               gr_i += 1
            elif comparator == Comparator.EQ:
               eq_feature_indices[eq_i] = condition.feature_index
               eq_thresholds[eq_i] = condition.threshold
               eq_i += 1
            else:
               neq_feature_indices[neq_i] = condition.feature_index
               neq_thresholds[neq_i] = condition.threshold
               neq_i += 1

            postincrement(iterator)

        cdef ConjunctiveBody body = ConjunctiveBody.__new__(ConjunctiveBody, leq_feature_indices, leq_thresholds,
                                                            gr_feature_indices, gr_thresholds, eq_feature_indices,
                                                            eq_thresholds, neq_feature_indices, neq_thresholds)
        cdef Head head

        if label_indices is None:
            head = FullHead.__new__(FullHead, scores)
        else:
            head = PartialHead.__new__(PartialHead, label_indices, scores)

        cdef rule = Rule.__new__(Rule, body, head)
        cdef RuleList rule_list = self.rule_list
        rule_list.add_rule(rule)

    cdef RuleModel build_model(self):
        cdef RuleList rule_list = self.rule_list
        cdef Rule default_rule = self.default_rule

        if default_rule is not None:
            rule_list.add_rule(default_rule)

        self.rule_list = None
        self.default_rule = None
        return rule_list
