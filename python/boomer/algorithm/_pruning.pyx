# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for pruning classification rules.
"""
from boomer.algorithm._arrays cimport array_intp
from boomer.algorithm._utils cimport test_condition
from boomer.algorithm._losses cimport Prediction

from cython.operator cimport dereference, postincrement


cdef class Pruning:
    """
    A base class for all classes that implement a strategy for pruning classification rules based on a "prune set",
    i.e., based on the examples that are not contained in the sub-sample that has been used to grow the rule (referred
    to as the "grow set").
    """

    cdef begin_pruning(self, uint32[::1] weights, Loss loss, HeadRefinement head_refinement,
                       intp[::1] covered_example_indices, intp[::1] label_indices):
        """
        Calculates the quality score of an existing rule, based on the examples that are contained in the prune set,
        i.e., based on all examples whose weight is 0.

        This function must be called prior to calling any other function provided by this class. It calculates and
        caches the original quality score of the existing rule before it is pruned. When invoking the function `prune`
        afterwards, the rule is pruned by removing individual conditions in a way that improves over the original
        quality score, if possible.

        :param weights:                 An array of dtype int, shape `(num_examples)`, representing the weights of all
                                        training examples, regardless of whether they are included in the prune set or
                                        grow set
        :param loss:                    The `Loss` to be minimized
        :param head_refinement:         The strategy that is used to find the heads of rules
        :param covered_example_indices: An array of dtype int, shape `(num_covered_examples)`, representing the indices
                                        of all training examples that are covered by the rule, regardless of whether
                                        they are included in the prune set or grow set
        :param label_indices:           An array of dtype int, shape `(num_predicted_labels)`, representing the indices
                                        of the labels for which the rule predicts or None, if the rule predicts for all
                                        labels
        """
        pass

    cdef intp[::1] prune(self, float32[::1, :] x, intp[::1, :] x_sorted_indices, list[s_condition] conditions):
        """
        Prunes the conditions of a rule by modifying a given list of conditions in-place.

        :param x:                   An array of dtype float, shape `(num_examples, num_features)`, representing the
                                    features of all training examples, regardless of whether they are included in the
                                    prune set or grow set
        :param x_sorted_indices:    An array of dtype int, shape `(num_examples, num_features)`, representing the
                                    indices of all training examples, regardless of whether they are included in the
                                    prune set or grow set, when sorting column-wise
        :param conditions:          A list that contains the rule's conditions
        :return:                    An array of dtype int, shape `(num_covered_examples)`, representing the indices of
                                    all training examples that are covered by the pruned rule, regardless of whether
                                    they are included in the prune set or grow set
        """
        pass


cdef class IREP(Pruning):
    """
    Implements incremental reduced error pruning (IREP) for pruning classification rules based on a "prune set".

    Given a rule with n conditions, IREP allows to remove up to n - 1 trailing conditions, depending on which of the
    pruning candidates improves the most over the overall quality score of the original rule (calculated on the prune
    set).
    """

    cdef begin_pruning(self, uint32[::1] weights, Loss loss, HeadRefinement head_refinement,
                       intp[::1] covered_example_indices, intp[::1] label_indices):
        cdef uint32 weight
        cdef intp i

        # Reset the loss function...
        loss.begin_search(label_indices)

        # Tell the loss function about all examples in the prune set that are covered by the given rule...
        for i in covered_example_indices:
            weight = weights[i]

            if weight == 0:
                loss.update_search(i, 1)

        # Calculate the optimal scores to be predicted by the given rule, as well as its overall quality score,  based
        # on the prune set...
        cdef Prediction prediction = head_refinement.evaluate_predictions(loss, 0)

        # Cache the overall quality score of the given rule based on the prune set...
        cdef float64 original_quality_score = prediction.overall_quality_score
        self.original_quality_score = original_quality_score

        # Cache arguments that will be used in the `prune` function...
        self.label_indices = label_indices
        self.covered_example_indices = covered_example_indices
        self.loss = loss
        self.head_refinement = head_refinement
        self.weights = weights

    cdef intp[::1] prune(self, float32[::1, :] x, intp[::1, :] x_sorted_indices, list[s_condition] conditions):
        cdef intp[::1] label_indices = self.label_indices
        cdef Loss loss = self.loss
        cdef HeadRefinement head_refinement = self.head_refinement
        cdef uint32[::1] weights = self.weights
        cdef intp num_conditions = conditions.size()
        cdef intp num_examples = x_sorted_indices.shape[0]
        cdef float64 best_quality_score = self.original_quality_score
        cdef intp[::1] best_covered_example_indices = self.covered_example_indices
        cdef intp best_num_examples = best_covered_example_indices.shape[0]
        cdef intp num_pruned_conditions = 0
        cdef list[s_condition].iterator iterator = conditions.begin()
        cdef s_condition condition
        cdef float32 threshold, feature_value
        cdef bint leq
        cdef intp[::1] covered_example_indices, new_covered_example_indices
        cdef uint32 weight
        cdef float64 quality_score
        cdef Prediction prediction
        cdef intp n, c, r, i, index, num_labels

        # We process the original rule's conditions (except for the last one) in the order they have been learned. At
        # each iteration we calculate the overall quality score of a rule that only contains the conditions processed so
        # far and keep track of the best one...
        for n in range(num_conditions - 1):
            condition = dereference(iterator)
            c = condition.feature_index
            threshold = condition.threshold
            leq = condition.leq

            # Reset the loss function...
            loss.begin_search(label_indices)

            # Initialize the array that contains the indices of the examples that satisfy the current condition (and all
            # previously processed conditions). At this point we don't know how many examples are exactly covered. For
            # this reason, the array's size is set to largest possible value, which is `num_examples`. If fewer examples
            # are covered, only the leading elements are set, the remaining ones remain undefined.
            new_covered_example_indices = array_intp(num_examples)
            i = 0

            if n == 0:
                # For the first condition, we traverse the examples in the order of their feature values. The order of
                # traversing depends on the condition's operator. If the condition uses the <= operator, we traverse in
                # ascending order, i.e., we start with the example with the smallest feature value. If the condition
                # uses the > operator, we traverse in descending order, i.e., we start with the example with the largest
                # feature value.
                for r in (range(num_examples) if leq else range(num_examples - 1, -1, -1)):
                    index = x_sorted_indices[r, c]
                    feature_value = x[index, c]

                    if test_condition(threshold, leq, feature_value):
                        # If the example satisfies the condition, we remember its index...
                        new_covered_example_indices[i] = index
                        i += 1

                        # If the example is contained in the prune set, i.e., if its weight is 0, we update the loss
                        # function...
                        weight = weights[index]

                        if weight == 0:
                            loss.update_search(index, 1)
                    else:
                        # If the example does not satisfy the condition, we are done, because the remaining ones will
                        # not satisfy the condition either...
                        break
            else:
                # For the remaining conditions we traverse the indices of the examples that satisfy all previously
                # processed conditions and check if they also satisfy the current one...
                for r in range(num_examples):
                    index = covered_example_indices[r]
                    feature_value = x[index, c]

                    if test_condition(threshold, leq, feature_value):
                        # If the example satisfies the condition, we remember its index...
                        new_covered_example_indices[i] = index
                        i += 1

                        # If the example is contained in the prune set, i.e., if its weight is 0, we update the loss
                        # function...
                        weight = weights[index]

                        if weight == 0:
                            loss.update_search(index, 1)

            # Update the number of covered examples (this is important, because otherwise we don't know how many of the
            # leading elements in `covered_example_indices` are set)...
            num_examples = i
            covered_example_indices = new_covered_example_indices

            # Calculate the optimal scores to be predicted by a rule that only contains the conditions processed so far,
            # as well as its overall quality score, based on the prune set...
            prediction = head_refinement.evaluate_predictions(loss, 0)

            # Check if the overall quality score of the current rule based on the prune set is better than the best
            # quality score known so far (reaching the same quality score with fewer conditions is also considered an
            # improvement)...
            quality_score = prediction.overall_quality_score

            if quality_score < best_quality_score or (num_pruned_conditions == 0 and quality_score <= best_quality_score):
                best_quality_score = quality_score
                best_covered_example_indices = covered_example_indices
                best_num_examples = num_examples
                num_pruned_conditions = num_conditions - (n + 1)

            postincrement(iterator)

        # Remove the pruned conditions...
        while num_pruned_conditions > 0:
            conditions.pop_back()
            num_pruned_conditions -= 1

        return best_covered_example_indices[0:best_num_examples]
