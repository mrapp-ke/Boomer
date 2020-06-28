"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for all (surrogate) loss functions to be minimized locally by the rules learned during training.
"""


cdef class Prediction:
    """
    Assesses the overall quality of a rule's predictions for one or several labels.
    """

    def __cinit__(self):
        self.predicted_scores = None
        self.overall_quality_score = 0


cdef class LabelIndependentPrediction(Prediction):
    """
    Assesses the quality of a rule's predictions for one or several labels independently from each other.
    """

    def __cinit__(self):
        self.quality_scores = None


cdef class Loss:
    """
    A base class for all (surrogate) loss functions to be minimized locally by the rules learned during training.

    An algorithm for rule induction may use the functions provided by this class to obtain loss-minimizing predictions
    for candidate rules (or the default rule), as well as quality scores that assess the quality of these rules.

    For reasons of efficiency, implementations of this class may be stateful. This enabled to avoid redundant
    recalculations of information that applies to several candidate rules. Call to functions of this class must follow a
    strict protocol regarding the order of function invocations. For detailed information refer to the documentation of
    the individual functions.
    """

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        """
        Calculates the loss-minimizing scores to be predicted by the default rule, i.e., a rule that covers all
        examples, for each label.

        This function must be called exactly once. It must be called prior to the invocation of any other function
        provided by this class.

        As this function is guaranteed to be invoked at first, it may be used to initialize the internal state of an
        instantiation of this class, i.e., to compute and store global information that is required by the other
        functions that will be called later, e.g. overall statistics about the given ground truth labels.

        :param y:   An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the training
                    examples according to the ground truth
        :return:    An array of dtype float, shape `(num_labels)`, representing the scores to be predicted by the
                    default rule for each label
        """
        pass

    cdef void begin_instance_sub_sampling(self):
        """
        Notifies the loss function that the examples, which should be considered in the following for learning a new
        rule or refining an existing one, have changed. The indices of the respective examples must be provided via
        subsequent calls to the function `update_sub_sample`.

        This function must be invoked before a new rule is learned from scratch (as each rule may be learned on a
        different sub-sample of the training data), as well as each time an existing rule has been refined, i.e.
        when a new condition has been added to its body (because this results in fewer examples being covered by the
        refined rule).

        This function is supposed to reset any non-global internal state that only holds for a certain set of examples
        and therefore becomes invalid when different examples are used, e.g. statistics about the ground truth labels of
        particular examples.
        """
        pass

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove):
        """
        Notifies the loss function about an example that should be considered in the following for learning a new rule
        or refining an existing one.

        This function must be called repeatedly for each example that should be considered, e.g., for all examples that
        have been selected via instance sub-sampling, immediately after the invocation of the function
        `begin_instance_sub_sampling`.

        Alternatively, this function may be used to indicate that an example, which has previously been passed to this
        function, should not be considered anymore by setting the argument `remove` accordingly.

        This function is supposed to update any internal state that relates to the considered examples, i.e., to compute
        and store local information that is required by the other functions that will be called later, e.g. statistics
        about the ground truth labels of these particular examples. Any information computed by this function is
        expected to be reset when invoking the function `begin_instance_sub_sample` for the next time.

        :param example_index:   The index of an example that should be considered
        :param weight:          The weight of the example that should be considered
        :param remove:          0, if the example should be considered, 1, if the example should not be considered
                                anymore
        """
        pass

    cdef void begin_search(self, intp[::1] label_indices):
        """
        Notifies the loss function that a new search for the best refinement of a rule, i.e., the best condition to be
        added to its body, should be started. The examples that are covered by such a condition must be provided via
        subsequent calls to the function `update_search`.

        This function must be called each time a new condition is considered, unless the new condition covers all
        examples previously provided via calls to the function `update_search`.

        This function is supposed to reset any internal state that only holds for the examples covered by a previously
        considered condition and therefore becomes invalid when different examples are covered by another condition,
        e.g. statistics about the ground truth labels of the covered examples.

        Optionally, a subset of the available labels may be specified via the argument `label_indices`. In such case,
        only the specified labels will be considered by the functions that will be called later. When calling this
        function again, a different set of labels may be specified.

        :param label_indices: An array of dtype int, shape `(num_predicted_labels)`, representing the indices of the
                              labels for which the refined rule should predict or None, if the rule may predict for all
                              labels
        """
        pass

    cdef void update_search(self, intp example_index, uint32 weight):
        """
        Notifies the loss function about an example that is covered by the condition that is currently considered for
        refining a rule.

        This function must be called repeatedly for each example that is covered by the current condition, immediately
        after the invocation of the function `begin_search`. Each of these examples must have been provided earlier via
        the function `update_sub_sample`.

        This function is supposed to update any internal state that relates to the examples that are covered current
        condition, i.e., to compute and store local information that is required by the other functions that will be
        called later, e.g. statistics about the ground truth labels of the covered examples. Any information computed by
        this function is expected to be reset when invoking the function `begin_search` or `reset_search` for the next
        time.

        :param example_index:   The index of the covered example
        :param weight:          The weight of the covered example
        """
        pass

    cdef void reset_search(self):
        """
        Resets the internal state that has been updated by preceding calls to the `update_search` function to the state
        after the `begin_search` function was called for the last time. Unlike a call to the `begin_search` function,
        which has the same effect, the current state is not purged entirely, but it is cached and made available for use
        by the functions `evaluate_label_dependent_predictions` and `evaluate_label_independent_predictions` (if the
        function argument `accumulated` is set accordingly).

        The information that is cached by this function is expected to be reset when the function `begin_search` is
        called for the next time. Before that, this function may be invoked multiple times (with one or several calls to
        `update_search` in between), which is supposed to update the previously cached state by accumulating the new
        one, i.e., when calling `begin_search` for the next time, the accumulated cached state should be the same as if
        `reset_search` would not have been called at all.
        """
        pass

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered, bint accumulated):
        """
        Calculates and returns the loss-minimizing scores to be predicted by a rule that covers all examples that have
        been provided so far via the function `update_search`.

        If the argument `uncovered` is 1, the rule is considered to cover all examples that belong to the difference
        between the examples that have been provided via the function `update_sub_sample` and the examples that have
        been provided via the function `update_search`.

        If the argument `accumulated` is 1, all examples that have been provided since the last call to the function
        `begin_search` are taken into account even if the function `reset_search` has been called before. If the latter
        has not been invoked, the argument does not have any effect.

        The calculated scores correspond to the subset of labels provided via the function `begin_search`. The score to
        be predicted for an individual label is calculated independently from the other labels, i.e., in case of a
        non-decomposable loss function, it is assumed that the rule will abstain for the other labels. In addition to
        each score, a quality score, which assesses the quality of the prediction for the respective label, is returned.

        :param uncovered:   0, if the rule covers all examples that have been provided via the function `update_search`,
                            1, if the rule covers all examples that belong to the difference between the examples that
                            have been provided via the function `update_sub_sample` and the examples that have been
                            provided via the function `update_search`
        :param accumulated: 0, if the rule covers all examples that have been provided via the function `update_search`
                            since the function `reset_search` has been called for the last time, 1, if the rule covers
                            all examples that have been provided since the last call to the function `begin_search`
        :return:            A `LabelIndependentPrediction` that stores the scores to be predicted by the rule for each
                            considered label, as well as the corresponding quality scores
        """
        pass

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered, bint accumulated):
        """
        Calculates and returns the loss-minimizing scores to be predicted by a rule that covers all examples that have
        been provided so far via the function `update_search`.

        If the argument `uncovered` is 1, the rule is considered to cover all examples that belong to the difference
        between the examples that have been provided via the function `update_sub_sample` and the examples that have
        been provided via the function `update_search`.

        If the argument `accumulated` is 1, all examples that have been provided since the last call to the function
        `begin_search` are taken into account even if the function `reset_search` has been called before. If the latter
        has not been invoked, the argument does not have any effect.

        The calculated scores correspond to the subset of labels provided via the function `begin_search`. The score to
        be predicted for an individual label is calculated with respect to the predictions for the other labels. In case
        of a decomposable loss function, i.e., if the labels are considered independently from each other, this function
        is equivalent to the function `evaluate_label_independent_predictions`. In addition to the scores, an overall
        quality score, which assesses the quality of the predictions for all labels in terms of a single score, is
        returned.

        :param uncovered:   0, if the rule covers all examples that have been provided via the function `update_search`,
                            1, if the rule covers all examples that belong to the difference between the examples that
                            have been provided via the function `update_sub_sample` and the examples that have been
                            provided via the function `update_search`
        :param accumulated: 0, if the rule covers all examples that have been provided via the function `update_search`
                            since the function `reset_search` has been called for the last time, 1, if the rule covers
                            all examples that have been provided since the last call to the function `begin_search`
        :return:            A `Prediction` that stores the optimal scores to be predicted by the rule for each
                            considered label, as well as its overall quality score
        """
        pass

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores):
        """
        Notifies the loss function about the predictions of a new rule that has been induced.

        This function must be called for each example that is covered by the new rule before learning the next rule,
        i.e., prior to the next invocation of the function `begin_instance_sub_sampling`.

        This function is supposed to update any internal state that depends on the predictions of already induced rules.

        :param example_index:       The index of an example that is covered by the newly induced rule, regardless of
                                    whether it is contained in the sub-sample or not
        :param label_indices:       An array of dtype int, shape `(num_predicted_labels)`, representing the indices of
                                    the labels for which the newly induced rule predicts or None, if the rule predicts
                                    for all labels
        :param predicted_scores:    An array of dtype float, shape `(num_predicted_labels)`, representing the scores
                                    that are predicted by the newly induced rule
        """
        pass
