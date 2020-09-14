"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for pruning classification rules.
"""
from boomer.common._arrays cimport float32, float64, array_uint32
from boomer.common._predictions cimport PredictionCandidate
from boomer.common._tuples cimport IndexedFloat32
from boomer.common.rules cimport Comparator
from boomer.common.statistics cimport AbstractRefinementSearch

from libcpp.memory cimport unique_ptr

from cython.operator cimport dereference, postincrement


cdef class Pruning:
    """
    A base class for all classes that implement a strategy for pruning classification rules based on a "prune set",
    i.e., based on the examples that are not contained in the sub-sample that has been used to grow the rule (referred
    to as the "grow set").
    """

    cdef pair[uint32[::1], uint32] prune(self, unordered_map[uint32, IndexedFloat32Array*]* sorted_feature_values_map,
                                         double_linked_list[Condition] conditions, Prediction* head,
                                         uint32[::1] covered_examples_mask, uint32 covered_examples_target,
                                         uint32[::1] weights, AbstractStatistics* statistics,
                                         HeadRefinement head_refinement):
        """
        Prunes the conditions of an existing rule by modifying a given list of conditions in-place. The rule is pruned
        by removing individual conditions in a way that improves over its original quality score as measured on the
        "prune set".

        :param sorted_feature_values_map:   A pointer to a map that maps feature indices to structs of type
                                            `IndexedFloat32Array`, storing the indices of all training examples, as well
                                            as their values for the respective feature, sorted in ascending order by the
                                            feature values
        :param conditions:                  A list that contains the conditions of the existing rule
        :param head:                        A pointer to an object of type `Prediction` representing the head of the
                                            existing rule
        :param covered_examples_mask:       An array of type `uint32`, shape `(num_examples)` that is used to keep track
                                            of the indices of the examples that are covered by the existing rule
        :param covered_examples_target:     The value that is used to mark those elements in `covered_examples_mask`
                                            that are covered by the existing rule
        :param weights:                     An array of type `uint32`, shape `(num_examples)`, representing the weights
                                            of all training examples, regardless of whether they are included in the
                                            prune set or grow set
        :param statistics:                  A pointer to an object of type `AbstractStatistics`, which served as the
                                            basis for learning the existing rule
        :param head_refinement:             The strategy that is used to find the heads of rules
        """
        pass


cdef class IREP(Pruning):
    """
    Implements incremental reduced error pruning (IREP) for pruning classification rules based on a "prune set".

    Given a rule with n conditions, IREP allows to remove up to n - 1 trailing conditions, depending on which of the
    pruning candidates improves the most over the overall quality score of the original rule (calculated on the prune
    set).
    """

    cdef pair[uint32[::1], uint32] prune(self, unordered_map[uint32, IndexedFloat32Array*]* sorted_feature_values_map,
                                         double_linked_list[Condition] conditions, Prediction* head,
                                         uint32[::1] covered_examples_mask, uint32 covered_examples_target,
                                         uint32[::1] weights, AbstractStatistics* statistics,
                                         HeadRefinement head_refinement):
        # The total number of training examples
        cdef uint32 num_examples = covered_examples_mask.shape[0]
        # The number of conditions of the existing rule
        cdef uint32 num_conditions = conditions.size()
        # The number of labels for which the existing rule predicts
        cdef uint32 num_predictions = head.numPredictions_
        # An array that stores the indices of the labels for which the existing rule predicts
        cdef uint32* label_indices = head.labelIndices_
        # Temporary variables
        cdef unique_ptr[AbstractRefinementSearch] refinement_search_ptr
        cdef PredictionCandidate* prediction
        cdef Condition condition
        cdef Comparator comparator
        cdef float32 threshold
        cdef float64 current_quality_score
        cdef IndexedFloat32Array* indexed_array
        cdef IndexedFloat32* indexed_values
        cdef uint32 feature_index, num_indexed_values, i, n, r, start, end
        cdef bint uncovered

        # Reset the statistics and start a new search...
        statistics.resetSampledStatistics()
        refinement_search_ptr.reset(statistics.beginSearch(num_predictions, label_indices))

        # Tell the statistics about all examples in the prune set that are covered by the existing rule...
        for i in range(num_examples):
            if weights[i] == 0:
                statistics.addSampledStatistic(i, 1)

                if covered_examples_mask[i] == covered_examples_target:
                    refinement_search_ptr.get().updateSearch(i, 1)

        # Determine the optimal prediction of the existing rule, as well as the corresponding quality score, based on
        # the prune set...
        prediction = head_refinement.calculate_prediction(refinement_search_ptr.get(), False, False)

        # Initialize variables that are used to keep track of the best rule...
        cdef float64 best_quality_score = prediction.overallQualityScore_
        cdef uint32[::1] best_covered_examples_mask = covered_examples_mask
        cdef uint32 best_covered_examples_target = covered_examples_target
        cdef uint32 num_pruned_conditions = 0

        # Initialize array that is used to keep track of the examples that are covered by the current rule...
        cdef uint32[::1] current_covered_examples_mask = array_uint32(num_examples)
        current_covered_examples_mask[:] = 0
        cdef uint32 current_covered_examples_target = 0

        # We process the existing rule's conditions (except for the last one) in the order they have been learned. At
        # each iteration, we calculate the quality score of a rule that only contains the conditions processed so far
        # and keep track of the best rule...
        cdef double_linked_list[Condition].iterator iterator = conditions.begin()

        for n in range(1, num_conditions):
            # Obtain properties of the current condition...
            condition = dereference(iterator)
            feature_index = condition.feature_index
            threshold = condition.threshold
            comparator = condition.comparator

            # Obtain the example indices and corresponding feature values for the feature, the current condition
            # corresponds to...
            indexed_array = dereference(sorted_feature_values_map)[feature_index]
            indexed_values = dereference(indexed_array).data
            num_indexed_values = dereference(indexed_array).numElements

            # Start a new search when processing a new condition...
            refinement_search_ptr.reset(statistics.beginSearch(num_predictions, label_indices))

            # Find the range [start, end) that either contains all covered or uncovered examples...
            end = __upper_bound(indexed_values, num_indexed_values, threshold)

            if comparator == Comparator.EQ or comparator == Comparator.NEQ:
                start = __lower_bound(indexed_values, end, threshold)

                if end - start == 0:
                    start = 0
                    end = num_indexed_values
                    uncovered = (comparator == Comparator.EQ)
                else:
                    uncovered = (comparator == Comparator.NEQ)
            else:
                if indexed_values[end].value > 0:
                    start = end
                    end = num_indexed_values
                    uncovered = (comparator == Comparator.LEQ)
                else:
                    start = 0
                    uncovered = (comparator == Comparator.GR)

            # Tell the statistics about the examples in range [start, end)...
            for r in range(start, end):
                i = indexed_values[r].index

                # We must only consider examples that are currently covered and contained in the prune set...
                if current_covered_examples_mask[i] == current_covered_examples_target and weights[i] == 0:
                    refinement_search_ptr.get().updateSearch(i, 1)

            # Check if the quality score of the current rule is better than the best quality score known so far
            # (reaching the same quality score with fewer conditions is also considered an improvement)...
            prediction = head_refinement.calculate_prediction(refinement_search_ptr.get(), uncovered, False)
            current_quality_score = prediction.overallQualityScore_

            if current_quality_score < best_quality_score or (num_pruned_conditions == 0 and current_quality_score <= best_quality_score):
                best_quality_score = current_quality_score
                best_covered_examples_mask[:] = current_covered_examples_mask
                best_covered_examples_target = current_covered_examples_target
                num_pruned_conditions = (num_conditions - n)

            # If at least one condition remains to be processed, we must update the array that is used to keep track of
            # the covered examples and notify the statistics about the updated sub-sample...
            if (n + 1) < num_conditions:
                if not uncovered:
                    statistics.resetCoveredStatistics()
                    current_covered_examples_target = n

                for r in range(start, end):
                    i = indexed_values[r].index

                    if current_covered_examples_mask[i] == current_covered_examples_target and weights[i] == 0:
                        statistics.updateCoveredStatistic(i, 1, uncovered)
                        current_covered_examples_mask[i] = n

            postincrement(iterator)

        # Remove the pruned conditions...
        while num_pruned_conditions > 0:
            conditions.pop_back()
            num_pruned_conditions -= 1

        cdef pair[uint32[::1], uint32] result
        result.first = best_covered_examples_mask
        result.second = best_covered_examples_target
        return result


cdef inline uint32 __upper_bound(IndexedFloat32* indexed_values, uint32 num_indexed_values, float32 threshold):
    """
    Returns the index of the first example in `indexed_values` with feature value > threshold. If no such example is
    found, `num_indexed_values` is returned.

    :param indexed_values:      A pointer to a C-array of type `IndexedFloat32`, storing the indices and feature values 
                                of examples
    :param num_indexed_values:  The number of leading elements in `indexed_values` to be considered
    :param threshold:           The threshold
    :return:                    The index of the first example in `indexed_values` with feature value > threshold or
                                `num_indexed_values`, if no such example is found
    """
    cdef uint32 first = 0
    cdef uint32 last = num_indexed_values
    cdef uint32 pivot
    cdef float32 pivot_value

    while first < last:
        pivot = first + ((last - first) / 2)
        pivot_value = indexed_values[pivot].value

        if threshold >= pivot_value:
            first = pivot + 1
        else:
            last = pivot

    return first


cdef inline uint32 __lower_bound(IndexedFloat32* indexed_values, uint32 num_indexed_values, float32 threshold):
    """
    Returns the index of the first example in `indexed_values` with feature value >= threshold. If no such example is
    found, `num_indexed_values` is returned.

    :param indexed_values:      A pointer to a C-array of type `IndexedFloat32`, storing the indices and feature values 
                                of examples
    :param num_indexed_values:  The number of leading elements in `indexed_values` to be considered
    :param threshold:           The threshold
    :return:                    The index of the first example in `indexed_values` with feature value >= threshold or
                                `num_indexed_values`, if no such example is found
    """
    cdef uint32 first = 0
    cdef uint32 last = num_indexed_values
    cdef uint32 pivot
    cdef float32 pivot_value

    while first < last:
        pivot = first + ((last - first) / 2)
        pivot_value = indexed_values[pivot].value

        if threshold <= pivot_value:
            last = pivot
        else:
            first = pivot + 1

    return first
