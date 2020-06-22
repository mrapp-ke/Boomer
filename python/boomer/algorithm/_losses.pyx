# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides base classes for loss functions to be minimized during training.
"""


cdef class Prediction:
    """
    Assess the overall quality of a rule's predictions for one or several labels.
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
    A base class for all decomposable or non-decomposable loss functions. A loss function can be used to calculate the
    optimal scores to be predicted by rules for individual labels and covered examples.
    """

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        """
        Calculates the optimal scores to be predicted by the default rule for each label and example.

        This function must be called prior to calling any other function provided by this class. It calculates and
        caches the gradients and hessians based on the expected confidence scores and the scores predicted by the 
        default rule.

        Furthermore, this function also allocates arrays for storing the total sum of gradients or hessians for each 
        label. These arrays are later used to search for the optimal scores to be predicted by rules that cover all
        examples provided to the search, as well as by rules that do not cover these instances (but all other ones) at
        the same time instead of requiring two passes through the examples. 
        
        To initialize the arrays, before invoking the function `begin_search`, the function 
        `begin_instance_sub_sampling` must be called, followed by invocations of the function `update_sub_sample` for 
        each of the examples to be considered by the search.

        :param y:   An array of dtype float, shape `(num_examples, num_labels)`, representing the labels of the training
                    examples according to the ground truth
        :return:    An array of dtype float, shape `(num_labels)`, representing the optimal scores to be predicted by
                    the default rule for each label and example
        """
        pass

    cdef begin_instance_sub_sampling(self):
        """
        Resets the cached sum of gradients and hessians for each label to 0.
        
        This function must always be invoked before the functions `begin_instance_sub_sampling` and `begin_search`. It 
        must also  be invoked when a rule has been refined, i.e., when a new condition has been added to its body, as 
        this results in less examples being covered by the rule.
        """
        pass

    cdef update_sub_sample(self, intp example_index):
        """
        Updates the total sum of gradients and hessians for each label given an example that should be considered by the 
        upcoming search. This should include all examples that have been selected via instance-sub-sampling and are 
        covered by the current rule (if any).

        This function must be invoked for each example that should be considered by the upcoming search after the 
        function `begin_instance_sub_sampling' and before `begin_search`.

        :param example_index: The index of an example that should be considered by the upcoming search
        """
        pass

    cdef begin_search(self, intp[::1] label_indices):
        """
        Begins a new search to find the optimal scores to be predicted by candidate rules for individual labels.

        This function must be called prior to searching for the best refinement with respect to a certain attribute in
        order to reset the sums of gradients and hessians cached internally to calculate the optimal scores more 
        efficiently. Subsequent invocations of the function `update_search` can be used to update the cached values 
        afterwards based on a single, newly covered example.
        
        Invoking the function `evaluate_label_independent_predictions` or `evaluate_label_dependent_predictions` at any
        point of the search (`update_search` must be called at least once before!) will yield the optimal scores to be
        predicted by a rule that covers all examples given so far, as well as corresponding quality scores that measure
        the quality of such a rule's predictions.

        When a new rule has been induced, the function `apply_predictions` must be invoked in order to update the cached
        gradients (and hessians in case of a non-decomposable loss function).

        :param label_indices: An array of dtype int, shape `(num_predicted_labels)`, representing the indices of the
                              labels for which the rule should predict or None, if the rule may predict for all labels
        """
        pass

    cdef update_search(self, intp example_index, uint32 weight):
        """
        Updates the cached sums of gradients and hessians based on a single, newly covered example.

        Subsequent invocations of the function `evaluate_label_independent_predictions` or
        `evaluate_label_dependent_predictions` will yield the optimal scores to be predicted by a rule that covers all
        examples given so far, as well as corresponding quality scores that measure the quality of such a rule's
        predictions.

        :param example_index:   The index of the newly-covered example in the entire training data set
        :param weight:          The weight of the newly covered example
        """
        pass

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered):
        """
        Calculates the optimal scores to be predicted by a rule that covers all examples that have been provided so far
        via the function `update_search`, respectively by a rule that covers all examples that have not been provided
        yet. Additionally, quality scores that measure the quality of the predicted scores are calculated for each
        label.

        The optimal score to be predicted for an individual label is calculated independently from the other labels. In
        case of a non-decomposable loss function, it is assumed that the rule will abstain, i.e., predict 0, for the
        other labels. The same assumption is used to calculate a quality score for each label independently.

        The calculated scores correspond to the label indices provided to the `begin_search` function. If no label
        indices were provided, scores for all labels are calculated.

        :param uncovered:   0, if the scores for a rule that covers all examples that have been provided so far should
                            be calculated, 1, if the scores for a rule that covers all examples that have not been
                            provided yet should be calculated
        :return:            A `LabelIndependentPrediction` that stores the optimal scores to be predicted by the rule,
                            as well as the corresponding quality scores for each label
        """
        pass

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered):
        """
        Calculates the optimal scores to be predicted by a rule that covers all examples that have been provided so far
        via the function `update_search`, respectively by a rule that covers all examples that have not been provided
        yet. Additionally, a single quality score that measures the quality of the predicted scores is calculated.

        The optimal score to be predicted for an individual label is calculated with respect to the predictions for the
        other labels. In case of a decomposable loss function, i.e., if the labels are independent from each other, the
        optimal scores provided by the function `evaluate_label_independent_predictions` are the same.

        The calculated scores correspond to the label indices provided to the `begin_search` function. If no label
        indices were provided, scores for all labels are calculated.

        :param uncovered:   0, if the scores for a rule that covers all examples that have been provided so far should
                            be calculated, 1, if the scores for a rule that covers all examples that have not been
                            provided yet should be calculated
        :return:            A `Prediction` that stores the optimal scores to be predicted by the rule, as well as its
                            overall quality score
        """
        pass

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        """
        Updates the cached gradients and hessians based on the predictions provided by newly induced rules.

        :param covered_example_indices: An array of dtype int, shape `(num_covered_examples)`, representing the indices
                                        of the examples that are covered by the newly induced rule, regardless of
                                        whether they are contained in the sub-sample or not
        :param label_indices:           An array of dtype int, shape `(num_predicted_labels)`, representing the indices
                                        of the labels for which the newly induced rule predicts or None, if the rule
                                        predicts for all labels
        :param predicted_scores:        An array of dtype float, shape `(num_predicted_labels)`, representing the scores
                                        that are predicted by the newly induced rule
        """
        pass

cdef class DecomposableLoss(Loss):
    """
    A base class for all decomposable loss functions.
    """

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        pass

    cdef begin_instance_sub_sampling(self):
        pass

    cdef update_sub_sample(self, intp example_index):
        pass

    cdef begin_search(self, intp[::1] label_indices):
        pass

    cdef update_search(self, intp example_index, uint32 weight):
        pass

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered):
        pass

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered):
        # In case of a decomposable loss, the label-dependent predictions are the same as the label-independent
        # predictions...
        return self.evaluate_label_independent_predictions(uncovered)

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        pass


cdef class NonDecomposableLoss(Loss):
    """
    A base class for all non-decomposable loss functions.
    """

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y):
        pass

    cdef begin_instance_sub_sampling(self):
        pass

    cdef update_sub_sample(self, intp example_index):
        pass

    cdef begin_search(self, intp[::1] label_indices):
        pass

    cdef update_search(self, intp example_index, uint32 weight):
        pass

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered):
        pass

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered):
        pass

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores):
        pass
