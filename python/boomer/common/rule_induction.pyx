"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement algorithms for inducing individual classification rules.
"""
from boomer.common._arrays cimport float64, array_uint32
from boomer.common._predictions cimport Prediction, PredictionCandidate
from boomer.common.rules cimport Condition, Comparator
from boomer.common.statistics cimport AbstractStatistics, AbstractRefinementSearch

from libc.math cimport fabs
from libc.stdlib cimport abs, malloc, realloc, free

from libcpp.list cimport list as double_linked_list
from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr

from cython.operator cimport dereference, postincrement
from cython.parallel cimport prange


"""
A struct that represents a potential refinement of a rule.
"""
cdef struct Refinement:
    PredictionCandidate* head
    uint32 feature_index
    float32 threshold
    Comparator comparator
    bint covered
    uint32 covered_weights
    intp start
    intp end
    intp previous
    IndexedFloat32Array* indexed_array
    IndexedFloat32ArrayWrapper* indexed_array_wrapper


cdef class RuleInduction:
    """
    A base class for all classes that implement an algorithm for the induction of individual classification rules.
    """

    cdef void induce_default_rule(self, StatisticsProvider statistics_provider, HeadRefinement head_refinement,
                                  ModelBuilder model_builder):
        """
        Induces the default rule.

        :param statistics_provider: A `StatisticsProvider` that provides access to the statistics which should serve as
                                    the basis for inducing the default rule
        :param head_refinement:     The strategy that should be used to find the head of the default rule or None, if no
                                    default rule should be used
        :param model_builder:       The builder, the default rule should be added to
        """
        pass

    cdef bint induce_rule(self, StatisticsProvider statistics_provider, uint8[::1] nominal_attribute_mask,
                          FeatureMatrix feature_matrix, HeadRefinement head_refinement,
                          LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                          FeatureSubSampling feature_sub_sampling, Pruning pruning, PostProcessor post_processor,
                          uint32 min_coverage, intp max_conditions, intp max_head_refinements, int num_threads, RNG rng,
                          ModelBuilder model_builder):
        """
        Induces a new classification rule.

        :param statistics_provider:     A `StatisticsProvider` that provides access to the statistics which should serve
                                        as the basis for inducing the new rule
        :param nominal_attribute_mask:  An array of type `uint8`, shape `(num_features)`, indicating whether the feature
                                        at a certain index is nominal (1) or not (0)
        :param feature_matrix:          A `FeatureMatrix` that provides column-wise access to the feature values of the
                                        training examples
        :param head_refinement:         The strategy that is used to find the heads of rules
        :param label_sub_sampling:      The strategy that should be used to sub-sample the labels or None, if no label
                                        sub-sampling should be used
        :param instance_sub_sampling:   The strategy that should be used to sub-sample the training examples or None, if
                                        no instance sub-sampling should be used
        :param feature_sub_sampling:    The strategy that should be used to sub-sample the available features or None,
                                        if no feature sub-sampling should be used
        :param pruning:                 The strategy that should be used to prune rules or None, if no pruning should be
                                        used
        :param post_processor:          The post-processor that should be used to post-process the rule once it has been
                                        learned or None, if no post-processing should be used
        :param min_coverage:            The minimum number of training examples that must be covered by the rule. Must
                                        be at least 1
        :param max_conditions:          The maximum number of conditions to be included in the rule's body. Must be at
                                        least 1 or -1, if the number of conditions should not be restricted
        :param max_head_refinements:    The maximum number of times the head of a rule may be refined after a new
                                        condition has been added to its body. Must be at least 1 or -1, if the number of
                                        refinements should not be restricted
        :param num_threads:             The number of threads to be used for evaluating the potential refinements of the
                                        rule in parallel. Must be at least 1
        :param rng:                     The random number generator to be used
        :param model_builder:           The builder, the rule should be added to
        :return:                        1, if a rule has been induced, 0 otherwise
        """
        pass


cdef class ExactGreedyRuleInduction(RuleInduction):
    """
    Allows to induce single- or multi-label classification rules using a greedy search, where new conditions are added
    iteratively to the (initially empty) body of a rule. At each iteration, the refinement that improves the rule the
    most is chosen. The search stops if no refinement results in an improvement. The possible conditions to be evaluated
    at each iteration result from an exact split finding algorithm, i.e., all possible thresholds that may be used by
    the conditions are considered.
    """

    def __cinit__(self):
        self.cache_global = new unordered_map[uint32, IndexedFloat32Array*]()

    def __dealloc__(self):
        cdef unordered_map[uint32, IndexedFloat32Array*]* cache_global = self.cache_global
        cdef unordered_map[uint32, IndexedFloat32Array*].iterator cache_global_iterator = cache_global.begin()
        cdef IndexedFloat32Array* indexed_array

        while cache_global_iterator != cache_global.end():
            indexed_array = dereference(cache_global_iterator).second
            free(indexed_array.data)
            free(indexed_array)
            postincrement(cache_global_iterator)

        del self.cache_global

    cdef void induce_default_rule(self, StatisticsProvider statistics_provider, HeadRefinement head_refinement,
                                  ModelBuilder model_builder):
        cdef unique_ptr[PredictionCandidate] default_prediction_ptr
        cdef unique_ptr[AbstractRefinementSearch] refinement_search_ptr
        cdef AbstractStatistics* statistics
        cdef uint32 num_statistics, i

        if head_refinement is not None:
            statistics = statistics_provider.get()
            num_statistics = statistics.numStatistics_
            statistics.resetSampledStatistics()

            for i in range(num_statistics):
                statistics.addSampledStatistic(i, 1)

            refinement_search_ptr.reset(statistics.beginSearch(0, NULL))
            default_prediction_ptr.reset(head_refinement.find_head(NULL, NULL, NULL, refinement_search_ptr.get(), True,
                                                                   False))

            statistics_provider.switch_rule_evaluation()

            for i in range(num_statistics):
                statistics.applyPrediction(i, default_prediction_ptr.get())

            model_builder.set_default_rule(default_prediction_ptr.get())
        else:
            statistics_provider.switch_rule_evaluation()

    cdef bint induce_rule(self, StatisticsProvider statistics_provider, uint8[::1] nominal_attribute_mask,
                          FeatureMatrix feature_matrix, HeadRefinement head_refinement,
                          LabelSubSampling label_sub_sampling, InstanceSubSampling instance_sub_sampling,
                          FeatureSubSampling feature_sub_sampling, Pruning pruning, PostProcessor post_processor,
                          uint32 min_coverage, intp max_conditions, intp max_head_refinements, int num_threads, RNG rng,
                          ModelBuilder model_builder):
        # The statistics
        cdef AbstractStatistics* statistics = statistics_provider.get()
        # The total number of statistics
        cdef uint32 num_statistics = statistics.numStatistics_
        # The total number of labels
        cdef uint32 num_labels = statistics.numLabels_
        # The total number of features
        cdef uint32 num_features = feature_matrix.num_features
        # A (stack-allocated) list that contains the conditions in the rule's body (in the order they have been learned)
        cdef double_linked_list[Condition] conditions
        # The total number of conditions
        cdef uint32 num_conditions = 0
        # An array representing the number of conditions per type of operator
        cdef uint32[::1] num_conditions_per_comparator = array_uint32(4)
        num_conditions_per_comparator[:] = 0
        # A map that stores the best refinement for each feature
        cdef unordered_map[uint32, Refinement] refinements  # Stack-allocated map
        # The best refinement of the current rule
        cdef Refinement best_refinement  # Stack-allocated struct
        best_refinement.head = NULL
        # Whether a refinement of the current rule has been found
        cdef bint found_refinement = True
        # An array that is used to keep track of the indices of the statistics are covered by the current rule. Each
        # element in the array corresponds to the statistic at the corresponding index. If the value for an element
        # is equal to `covered_statistics_target`, it is covered by the current rule, otherwise it is not.
        cdef uint32[::1] covered_statistics_mask = array_uint32(num_statistics)
        covered_statistics_mask[:] = 0
        cdef uint32 covered_statistics_target = 0
        # A map that stores potential thresholds that result from all available statistics
        cdef unordered_map[uint32, IndexedFloat32Array*]* cache_global = self.cache_global
        # A map that stores potential thresholds that result from the statistics that are covered by the current rule
        cdef unordered_map[uint32, IndexedFloat32ArrayWrapper*] cache_local  # Stack-allocated map

        # Temporary variables
        cdef unordered_map[uint32, IndexedFloat32ArrayWrapper*].iterator cache_local_iterator
        cdef IndexedFloat32ArrayWrapper* indexed_array_wrapper
        cdef IndexedFloat32Array* indexed_array
        cdef IndexedFloat32* indexed_values
        cdef Refinement current_refinement
        cdef uint32[::1] sampled_feature_indices
        cdef uint32 num_sampled_features, weight, f, r
        cdef bint nominal
        cdef intp c

        # Sub-sample examples, if necessary...
        cdef pair[uint32[::1], uint32] uint32_array_scalar_pair
        cdef uint32[::1] weights
        cdef uint32 total_sum_of_weights

        if instance_sub_sampling is None:
            weights = None
            total_sum_of_weights = num_statistics
        else:
            uint32_array_scalar_pair = instance_sub_sampling.sub_sample(num_statistics, rng)
            weights = uint32_array_scalar_pair.first
            total_sum_of_weights = uint32_array_scalar_pair.second

        # Notify the statistics about the examples that are included in the sub-sample...
        statistics.resetSampledStatistics()

        for r in range(num_statistics):
            weight = 1 if weights is None else weights[r]
            statistics.addSampledStatistic(r, weight)

        # Sub-sample labels, if necessary...
        cdef uint32[::1] sampled_label_indices
        cdef const uint32* label_indices
        cdef uint32 num_predictions

        if label_sub_sampling is None:
            sampled_label_indices = None
            label_indices = <const uint32*>NULL
            num_predictions = 0
        else:
            sampled_label_indices = label_sub_sampling.sub_sample(num_labels, rng)
            label_indices = &sampled_label_indices[0]
            num_predictions = sampled_label_indices.shape[0]

        try:
            # Search for the best refinement until no improvement in terms of the rule's quality score is possible
            # anymore or the maximum number of conditions has been reached...
            while found_refinement and (max_conditions == -1 or num_conditions < max_conditions):
                found_refinement = False

                # Sub-sample features, if necessary...
                if feature_sub_sampling is None:
                    sampled_feature_indices = None
                    num_sampled_features = num_features
                else:
                    sampled_feature_indices = feature_sub_sampling.sub_sample(num_features, rng)
                    num_sampled_features = sampled_feature_indices.shape[0]

                # For each feature, update the caches `cache_global` and 'cache_local`, if necessary...
                for c in range(num_sampled_features):
                    f = <uint32>c if sampled_feature_indices is None else sampled_feature_indices[c]
                    __update_caches(f, cache_global, cache_local)

                # Search for the best condition among all available features to be added to the current rule...
                for c in prange(num_sampled_features, nogil=True, schedule='dynamic', num_threads=num_threads):
                    f = <uint32>c if sampled_feature_indices is None else sampled_feature_indices[c]
                    nominal = nominal_attribute_mask is not None and nominal_attribute_mask[f] > 0
                    current_refinement = __find_refinement(f, nominal, num_predictions, label_indices, weights,
                                                           total_sum_of_weights, cache_global, cache_local,
                                                           feature_matrix, covered_statistics_mask,
                                                           covered_statistics_target, num_conditions, statistics,
                                                           head_refinement, best_refinement.head)

                    with gil:
                        refinements[f] = current_refinement

                # Pick the best refinement among the refinements that have been found for the different features...
                for c in range(num_sampled_features):
                    f = <uint32>c if sampled_feature_indices is None else sampled_feature_indices[c]
                    current_refinement = refinements[f]

                    if current_refinement.head != NULL and (best_refinement.head == NULL
                                                            or current_refinement.head.overallQualityScore_ < best_refinement.head.overallQualityScore_):
                        del best_refinement.head
                        best_refinement = current_refinement
                        found_refinement = True
                    else:
                        del current_refinement.head

                refinements.clear()

                if found_refinement:
                    # If a refinement has been found, add the new condition...
                    conditions.push_back(__make_condition(best_refinement.feature_index, best_refinement.comparator,
                                                          best_refinement.threshold))
                    num_conditions += 1
                    num_conditions_per_comparator[<uint32>best_refinement.comparator] += 1

                    if max_head_refinements > 0 and num_conditions >= max_head_refinements:
                        # Keep the labels for which the rule predicts, if the head should not be further refined...
                        num_predictions = best_refinement.head.numPredictions_
                        label_indices = best_refinement.head.labelIndices_

                    # If instance sub-sampling is used, examples that are not contained in the current sub-sample were
                    # not considered for finding the new condition. In the next step, we need to identify the examples
                    # that are covered by the refined rule, including those that are not contained in the sub-sample,
                    # via the function `__filter_current_indices`. Said function calculates the number of covered
                    # examples based on the variable `best_refinement.end`, which represents the position that separates
                    # the covered from the uncovered examples. However, when taking into account the examples that are
                    # not contained in the sub-sample, this position may differ from the current value of
                    # `best_refinement.end` and therefore must be adjusted...
                    if weights is not None and abs(best_refinement.previous - best_refinement.end) > 1:
                        best_refinement.end = __adjust_split(best_refinement.indexed_array, best_refinement.end,
                                                             best_refinement.previous, best_refinement.threshold)

                    # Identify the examples for which the rule predicts...
                    covered_statistics_target = __filter_current_indices(best_refinement.indexed_array,
                                                                         best_refinement.indexed_array_wrapper,
                                                                         best_refinement.start, best_refinement.end,
                                                                         best_refinement.comparator,
                                                                         best_refinement.covered, num_conditions,
                                                                         covered_statistics_mask,
                                                                         covered_statistics_target, statistics, weights)
                    total_sum_of_weights = best_refinement.covered_weights

                    if total_sum_of_weights <= min_coverage:
                        # Abort refinement process if the rule is not allowed to cover less examples...
                        break

            if best_refinement.head == NULL:
                # No rule could be induced, because no useful condition could be found. This is for example the case, if
                # all features are constant.
                return False
            else:
                if weights is not None:
                    # Prune rule, if necessary (a rule can only be pruned if it contains more than one condition)...
                    if pruning is not None and num_conditions > 1:
                        uint32_array_scalar_pair = pruning.prune(cache_global, conditions, best_refinement.head,
                                                                 covered_statistics_mask, covered_statistics_target,
                                                                 weights, statistics, head_refinement)
                        covered_statistics_mask = uint32_array_scalar_pair.first
                        covered_statistics_target = uint32_array_scalar_pair.second

                    # If instance sub-sampling is used, we need to re-calculate the scores in the head based on the
                    # entire training data...
                    __recalculate_predictions(statistics, num_statistics, head_refinement, covered_statistics_mask,
                                              covered_statistics_target, best_refinement.head)

                # Apply post-processor, if necessary...
                if post_processor is not None:
                    post_processor.post_process(best_refinement.head)

                # Update the statistics based on the predictions of the new rule...
                for r in range(num_statistics):
                    if covered_statistics_mask[r] == covered_statistics_target:
                        statistics.applyPrediction(r, best_refinement.head)

                # Add the induced rule to the model...
                model_builder.add_rule(best_refinement.head, conditions, num_conditions_per_comparator)
                return True
        finally:
            del best_refinement.head

            # Free memory occupied by the arrays stored in `cache_local`...
            cache_local_iterator = cache_local.begin()

            while cache_local_iterator != cache_local.end():
                indexed_array_wrapper = dereference(cache_local_iterator).second
                indexed_array = indexed_array_wrapper.array

                if indexed_array != NULL:
                    indexed_values = indexed_array.data
                    free(indexed_values)

                free(indexed_array)
                free(indexed_array_wrapper)
                postincrement(cache_local_iterator)


cdef void __update_caches(uint32 feature_index, unordered_map[uint32, IndexedFloat32Array*]* cache_global,
                          unordered_map[uint32, IndexedFloat32ArrayWrapper*] &cache_local):
    """
    Updates the caches `cache_global` and `cache_local`, which store arrays that contain the indices of examples, as
    well as their values for certain features, if necessary.

    :param feature_index:               The index of the feature, the new condition should correspond to
    :param cache_global:                A pointer to a map that maps feature indices to structs of type
                                        `IndexedFloat32Array`, storing the indices of all training examples, as well
                                        as their values for the respective feature, sorted in ascending order by the
                                        feature values
    :param cache_local:                 A pointer to a map that maps feature indices to structs of type
                                        `IndexedFloat32ArrayWrapper`, storing the indices of the training examples that
                                        are covered by the existing rule, as well as their values for the respective
                                        feature, sorted in ascending order by the feature values
    """
    cdef IndexedFloat32ArrayWrapper* indexed_array_wrapper = cache_local[feature_index]

    if indexed_array_wrapper == NULL:
        indexed_array_wrapper = <IndexedFloat32ArrayWrapper*>malloc(sizeof(IndexedFloat32ArrayWrapper))
        indexed_array_wrapper.array = NULL
        indexed_array_wrapper.num_conditions = 0
        cache_local[feature_index] = indexed_array_wrapper

    cdef IndexedFloat32Array* indexed_array = indexed_array_wrapper.array

    if indexed_array == NULL:
        indexed_array = dereference(cache_global)[feature_index]

        if indexed_array == NULL:
            indexed_array = <IndexedFloat32Array*>malloc(sizeof(IndexedFloat32Array))
            indexed_array.data = NULL
            indexed_array.numElements = 0
            dereference(cache_global)[feature_index] = indexed_array


cdef Refinement __find_refinement(uint32 feature_index, bint nominal, uint32 num_label_indices,
                                  const uint32* label_indices, uint32[::1] weights, uint32 total_sum_of_weights,
                                  unordered_map[uint32, IndexedFloat32Array*]* cache_global,
                                  unordered_map[uint32, IndexedFloat32ArrayWrapper*] &cache_local,
                                  FeatureMatrix feature_matrix, uint32[::1] covered_statistics_mask,
                                  uint32 covered_statistics_target, uint32 num_conditions,
                                  AbstractStatistics* statistics, HeadRefinement head_refinement,
                                  PredictionCandidate* head) nogil:
    """
    Finds and returns the best refinement of an existing rule, which results from adding a new condition that
    corresponds to a certain feature.

    :param feature_index:               The index of the feature, the new condition should correspond to
    :param nominal                      1, if the feature, the new condition should correspond to, is nominal, 0
                                        otherwise
    :param num_label_indices:           The number of elements in the array `label_indices`
    :param label_indices:               A pointer to an array of type `uint32`, shape `(num_predictions)`, representing
                                        the indices of the labels for which the refined rule may predict
    :param weights:                     An array of type `uint32`, shape `(num_statistics)`, representing the weights of
                                        the training examples or None, if all training examples are weighed equally
    :param total_sum_of_weights:        The sum of the weights of all training examples
    :param cache_global:                A pointer to a map that maps feature indices to structs of type
                                        `IndexedFloat32Array`, storing the indices of all training examples, as well as
                                        their values for the respective feature, sorted in ascending order by the
                                        feature values
    :param cache_local:                 A pointer to a map that maps feature indices to structs of type
                                        `IndexedFloat32ArrayWrapper`, storing the indices of the training examples that
                                        are covered by the existing rule, as well as their values for the respective
                                        feature, sorted in ascending order by the feature values
    :param feature_matrix:              A `FeatureMatrix` that provides column-wise access to the feature values of the
                                        training examples
    :param covered_statistics_mask:     An array of type `uint32`, shape `(num_statistics)` that is used to keep track
                                        of the indices of the statistics that are covered by the existing rule. It will
                                        be updated by this function
    :param covered_statistics_target:   The value that is used to mark those elements in `covered_statistics_mask` that
                                        are covered by the existing rule
    :param num_conditions:              The number of conditions in the body of the existing rule
    :param statistics:                  A pointer to an object of type `AbstractStatistics` to be used for finding the
                                        best refinement
    :param head_refinement:             The strategy that should be used to find the head of the refined rule
    :param head:                        A pointer to an object of type `PredictionCandidate`, representing the head of
                                        the existing rule
    :return:                            A struct of type `Refinement`, representing the best refinement that has been
                                        found
    """
    # The current refinement of the existing rule
    cdef Refinement refinement  # Stack-allocated struct
    refinement.feature_index = feature_index
    refinement.head = NULL
    # The best head seen so far
    cdef PredictionCandidate* best_head = head
    # Temporary variables
    cdef PredictionCandidate* current_head
    cdef float32 current_threshold, previous_threshold, previous_threshold_negative
    cdef uint32 weight, accumulated_sum_of_weights, accumulated_sum_of_weights_negative
    cdef uint32 total_accumulated_sum_of_weights, i, r, previous_r, previous_r_negative

    # Obtain array that contains the indices of the training examples sorted according to the current feature...
    cdef IndexedFloat32ArrayWrapper* indexed_array_wrapper = cache_local[feature_index]
    cdef IndexedFloat32Array* indexed_array = indexed_array_wrapper.array
    cdef IndexedFloat32* indexed_values

    if indexed_array == NULL:
        indexed_array = dereference(cache_global)[feature_index]
        indexed_values = indexed_array.data

        if indexed_values == NULL:
            feature_matrix.fetch_sorted_feature_values(feature_index, indexed_array)
            indexed_values = indexed_array.data

    # Filter indices, if only a subset of the contained examples is covered...
    if num_conditions > indexed_array_wrapper.num_conditions:
        __filter_any_indices(indexed_array, indexed_array_wrapper, num_conditions, covered_statistics_mask,
                             covered_statistics_target)
        indexed_array = indexed_array_wrapper.array

    cdef uint32 num_indexed_values = indexed_array.numElements
    indexed_values = indexed_array.data

    # Start a new search based on the current statistics when processing a new feature...
    cdef unique_ptr[AbstractRefinementSearch] refinement_search_ptr
    refinement_search_ptr.reset(statistics.beginSearch(num_label_indices, label_indices))

    # In the following, we start by processing all examples with feature values < 0...
    cdef uint32 sum_of_weights = 0
    cdef intp first_r = 0
    cdef intp last_negative_r = -1

    # Traverse examples with feature values < 0 in ascending order until the first example with weight > 0 is
    # encountered...
    for r in range(num_indexed_values):
        current_threshold = indexed_values[r].value

        if current_threshold >= 0:
            break

        last_negative_r = r
        i = indexed_values[r].index
        weight = 1 if weights is None else weights[i]

        if weight > 0:
            # Tell the search that the example will be covered by upcoming refinements...
            refinement_search_ptr.get().updateSearch(i, weight)
            sum_of_weights += weight
            previous_threshold = current_threshold
            previous_r = r
            break

    accumulated_sum_of_weights = sum_of_weights

    # Traverse the remaining examples with feature values < 0 in ascending order...
    if sum_of_weights > 0:
        for r in range(r + 1, num_indexed_values):
            current_threshold = indexed_values[r].value

            if current_threshold >= 0:
                break

            last_negative_r = r
            i = indexed_values[r].index
            weight = 1 if weights is None else weights[i]

            # Do only consider examples that are included in the current sub-sample...
            if weight > 0:
                # Split points between examples with the same feature value must not be considered...
                if previous_threshold != current_threshold:
                    # Find and evaluate the best head for the current refinement, if a condition that uses the <=
                    # operator (or the == operator in case of a nominal feature) is used...
                    current_head = head_refinement.find_head(best_head, refinement.head, label_indices,
                                                             refinement_search_ptr.get(), False, False)

                    # If the refinement is better than the current rule...
                    if current_head != NULL:
                        best_head = current_head
                        refinement.head = current_head
                        refinement.start = first_r
                        refinement.end = r
                        refinement.previous = previous_r
                        refinement.covered_weights = sum_of_weights
                        refinement.indexed_array = indexed_array
                        refinement.indexed_array_wrapper = indexed_array_wrapper
                        refinement.covered = True

                        if nominal:
                            refinement.comparator = Comparator.EQ
                            refinement.threshold = previous_threshold
                        else:
                            refinement.comparator = Comparator.LEQ
                            refinement.threshold = (previous_threshold + current_threshold) / 2.0

                    # Find and evaluate the best head for the current refinement, if a condition that uses the >
                    # operator (or the != operator in case of a nominal feature) is used...
                    current_head = head_refinement.find_head(best_head, refinement.head, label_indices,
                                                             refinement_search_ptr.get(), True, False)

                    # If the refinement is better than the current rule...
                    if current_head != NULL:
                        best_head = current_head
                        refinement.head = current_head
                        refinement.start = first_r
                        refinement.end = r
                        refinement.previous = previous_r
                        refinement.covered_weights = (total_sum_of_weights - sum_of_weights)
                        refinement.indexed_array = indexed_array
                        refinement.indexed_array_wrapper = indexed_array_wrapper
                        refinement.covered = False

                        if nominal:
                            refinement.comparator = Comparator.NEQ
                            refinement.threshold = previous_threshold
                        else:
                            refinement.comparator = Comparator.GR
                            refinement.threshold = (previous_threshold + current_threshold) / 2.0

                    # Reset the search in case of a nominal feature, as the previous examples will not be covered by the
                    # next condition...
                    if nominal:
                        refinement_search_ptr.get().resetSearch()
                        sum_of_weights = 0
                        first_r = r

                previous_threshold = current_threshold
                previous_r = r

                # Tell the search that the example will be covered by upcoming refinements...
                refinement_search_ptr.get().updateSearch(i, weight)
                sum_of_weights += weight
                accumulated_sum_of_weights += weight

        # If the feature is nominal and the examples that have been iterated so far do not all have the same feature
        # value, or if not all examples have been iterated so far, we must evaluate additional conditions
        # `f == previous_threshold` and `f != previous_threshold`...
        if nominal and sum_of_weights > 0 and (sum_of_weights < accumulated_sum_of_weights
                                               or accumulated_sum_of_weights < total_sum_of_weights):
            # Find and evaluate the best head for the current refinement, if a condition that uses the == operator is
            # used...
            current_head = head_refinement.find_head(best_head, refinement.head, label_indices,
                                                     refinement_search_ptr.get(), False, False)

            # If the refinement is better than the current rule...
            if current_head != NULL:
                best_head = current_head
                refinement.head = current_head
                refinement.start = first_r
                refinement.end = (last_negative_r + 1)
                refinement.previous = previous_r
                refinement.covered_weights = sum_of_weights
                refinement.indexed_array = indexed_array
                refinement.indexed_array_wrapper = indexed_array_wrapper
                refinement.covered = True
                refinement.comparator = Comparator.EQ
                refinement.threshold = previous_threshold

            # Find and evaluate the best head for the current refinement, if a condition that uses the != operator is
            # used...
            current_head = head_refinement.find_head(best_head, refinement.head, label_indices,
                                                     refinement_search_ptr.get(), True, False)

            # If the refinement is better than the current rule...
            if current_head != NULL:
                best_head = current_head
                refinement.head = current_head
                refinement.start = first_r
                refinement.end = (last_negative_r + 1)
                refinement.previous = previous_r
                refinement.covered_weights = (total_sum_of_weights - sum_of_weights)
                refinement.indexed_array = indexed_array
                refinement.indexed_array_wrapper = indexed_array_wrapper
                refinement.covered = False
                refinement.comparator = Comparator.NEQ
                refinement.threshold = previous_threshold

        # Reset the search, if any examples with feature value < 0 have been processed...
        refinement_search_ptr.get().resetSearch()

    previous_threshold_negative = previous_threshold
    previous_r_negative = previous_r
    accumulated_sum_of_weights_negative = accumulated_sum_of_weights

    # We continue by processing all examples with feature values >= 0...
    sum_of_weights = 0
    first_r = num_indexed_values - 1

    # Traverse examples with feature values >= 0 in descending order until the first example with weight > 0 is
    # encountered...
    for r in range(first_r, last_negative_r, -1):
        i = indexed_values[r].index
        weight = 1 if weights is None else weights[i]

        if weight > 0:
            # Tell the search that the example will be covered by upcoming refinements...
            refinement_search_ptr.get().updateSearch(i, weight)
            sum_of_weights += weight
            previous_threshold = indexed_values[r].value
            previous_r = r
            break

    accumulated_sum_of_weights = sum_of_weights

    # Traverse the remaining examples with feature values >= 0 in descending order...
    if sum_of_weights > 0:
        for r in range(r - 1, last_negative_r, -1):
            i = indexed_values[r].index
            weight = 1 if weights is None else weights[i]

            # Do only consider examples that are included in the current sub-sample...
            if weight > 0:
                current_threshold = indexed_values[r].value

                # Split points between examples with the same feature value must not be considered...
                if previous_threshold != current_threshold:
                    # Find and evaluate the best head for the current refinement, if a condition that uses the >
                    # operator (or the == operator in case of a nominal feature) is used...
                    current_head = head_refinement.find_head(best_head, refinement.head, label_indices,
                                                             refinement_search_ptr.get(), False, False)

                    # If the refinement is better than the current rule...
                    if current_head != NULL:
                        best_head = current_head
                        refinement.head = current_head
                        refinement.start = first_r
                        refinement.end = r
                        refinement.previous = previous_r
                        refinement.covered_weights = sum_of_weights
                        refinement.indexed_array = indexed_array
                        refinement.indexed_array_wrapper = indexed_array_wrapper
                        refinement.covered = True

                        if nominal:
                            refinement.comparator = Comparator.EQ
                            refinement.threshold = previous_threshold
                        else:
                            refinement.comparator = Comparator.GR
                            refinement.threshold = (previous_threshold + current_threshold) / 2.0

                    # Find and evaluate the best head for the current refinement, if a condition that uses the <=
                    # operator (or the != operator in case of a nominal feature) is used...
                    current_head = head_refinement.find_head(best_head, refinement.head, label_indices,
                                                             refinement_search_ptr.get(), True, False)

                    # If the refinement is better than the current rule...
                    if current_head != NULL:
                        best_head = current_head
                        refinement.head = current_head
                        refinement.start = first_r
                        refinement.end = r
                        refinement.previous = previous_r
                        refinement.covered_weights = (total_sum_of_weights - sum_of_weights)
                        refinement.indexed_array = indexed_array
                        refinement.indexed_array_wrapper = indexed_array_wrapper
                        refinement.covered = False

                        if nominal:
                            refinement.comparator = Comparator.NEQ
                            refinement.threshold = previous_threshold
                        else:
                            refinement.comparator = Comparator.LEQ
                            refinement.threshold = (previous_threshold + current_threshold) / 2.0

                    # Reset the search in case of a nominal feature, as the previous examples will not be covered by the
                    # next condition...
                    if nominal:
                        refinement_search_ptr.get().resetSearch()
                        sum_of_weights = 0
                        first_r = r

                previous_threshold = current_threshold
                previous_r = r

                # Tell the search that the example will be covered by upcoming refinements...
                refinement_search_ptr.get().updateSearch(i, weight)
                sum_of_weights += weight
                accumulated_sum_of_weights += weight

    # If the feature is nominal and the examples with feature values >= 0 that have been iterated so far do not all have
    # the same feature value, we must evaluate additional conditions `f == previous_threshold` and
    # `f != previous_threshold`...
    if nominal and sum_of_weights > 0 and sum_of_weights < accumulated_sum_of_weights:
        # Find and evaluate the best head for the current refinement, if a condition that uses the == operator is
        # used...
        current_head = head_refinement.find_head(best_head, refinement.head, label_indices, refinement_search_ptr.get(),
                                                 False, False)

        # If the refinement is better than the current rule...
        if current_head != NULL:
            best_head = current_head
            refinement.head = current_head
            refinement.start = first_r
            refinement.end = last_negative_r
            refinement.previous = previous_r
            refinement.covered_weights = sum_of_weights
            refinement.indexed_array = indexed_array
            refinement.indexed_array_wrapper = indexed_array_wrapper
            refinement.covered = True
            refinement.comparator = Comparator.EQ
            refinement.threshold = previous_threshold

        # Find and evaluate the best head for the current refinement, if a condition that uses the != operator is
        # used...
        current_head = head_refinement.find_head(best_head, refinement.head, label_indices, refinement_search_ptr.get(),
                                                 True, False)

        # If the refinement is better than the current rule...
        if current_head != NULL:
            best_head = current_head
            refinement.head = current_head
            refinement.start = first_r
            refinement.end = last_negative_r
            refinement.previous = previous_r
            refinement.covered_weights = (total_sum_of_weights - sum_of_weights)
            refinement.indexed_array = indexed_array
            refinement.indexed_array_wrapper = indexed_array_wrapper
            refinement.covered = False
            refinement.comparator = Comparator.NEQ
            refinement.threshold = previous_threshold

    total_accumulated_sum_of_weights = accumulated_sum_of_weights_negative + accumulated_sum_of_weights

    # If the sum of weights of all examples that have been iterated so far (including those with feature values < 0 and
    # those with feature values >= 0) is less than the sum of of weights of all examples, this means that there are
    # examples with sparse, i.e. zero, feature values. In such case, we must explicitly test conditions that separate
    # these examples from the ones that have already been iterated...
    if total_accumulated_sum_of_weights > 0 and total_accumulated_sum_of_weights < total_sum_of_weights:
        # If the feature is nominal, we must reset the search once again to ensure that the accumulated state includes
        # all examples that have been processed so far...
        if nominal:
            refinement_search_ptr.get().resetSearch()
            first_r = num_indexed_values - 1

        # Find and evaluate the best head for the current refinement, if the condition `f > previous_threshold / 2` (or
        # the condition `f != 0` in case of a nominal feature) is used...
        current_head = head_refinement.find_head(best_head, refinement.head, label_indices, refinement_search_ptr.get(),
                                                 False, nominal)

        # If the refinement is better than the current rule...
        if current_head != NULL:
            best_head = current_head
            refinement.head = current_head
            refinement.start = first_r
            refinement.indexed_array = indexed_array
            refinement.indexed_array_wrapper = indexed_array_wrapper
            refinement.covered = True

            if nominal:
                refinement.end = -1
                refinement.previous = -1
                refinement.covered_weights = total_accumulated_sum_of_weights
                refinement.comparator = Comparator.NEQ
                refinement.threshold = 0.0
            else:
                refinement.end = last_negative_r
                refinement.previous = previous_r
                refinement.covered_weights = accumulated_sum_of_weights
                refinement.comparator = Comparator.GR
                refinement.threshold = previous_threshold / 2.0

        # Find and evaluate the best head for the current refinement, if the condition `f <= previous_threshold / 2` (or
        # `f == 0` in case of a nominal feature) is used...
        current_head = head_refinement.find_head(best_head, refinement.head, label_indices, refinement_search_ptr.get(),
                                                 True, nominal)

        # If the refinement is better than the current rule...
        if current_head != NULL:
            best_head = current_head
            refinement.head = current_head
            refinement.start = first_r
            refinement.indexed_array = indexed_array
            refinement.indexed_array_wrapper = indexed_array_wrapper
            refinement.covered = False

            if nominal:
                refinement.end = -1
                refinement.previous = -1
                refinement.covered_weights = (total_sum_of_weights - total_accumulated_sum_of_weights)
                refinement.comparator = Comparator.EQ
                refinement.threshold = 0.0
            else:
                refinement.end = last_negative_r
                refinement.previous = previous_r
                refinement.covered_weights = (total_sum_of_weights - accumulated_sum_of_weights)
                refinement.comparator = Comparator.LEQ
                refinement.threshold = previous_threshold / 2.0

    # If the feature is numerical and there are other examples than those with feature values < 0 that have been
    # processed earlier, we must evaluate additional conditions that separate the examples with feature values < 0 from
    # the remaining ones (unlike in the nominal case, these conditions cannot be evaluated earlier, because it remains
    # unclear what the thresholds of the conditions should be until the examples with feature values >= 0 have been
    # processed).
    if not nominal and accumulated_sum_of_weights_negative > 0 and accumulated_sum_of_weights_negative < total_sum_of_weights:
        # Find and evaluate the best head for the current refinement, if the condition that uses the <= operator is
        # used...
        current_head = head_refinement.find_head(best_head, refinement.head, label_indices, refinement_search_ptr.get(),
                                                 False, True)

        if current_head != NULL:
            best_head = current_head
            refinement.head = current_head
            refinement.start = 0
            refinement.end = (last_negative_r + 1)
            refinement.previous = previous_r_negative
            refinement.covered_weights = accumulated_sum_of_weights_negative
            refinement.indexed_array = indexed_array
            refinement.indexed_array_wrapper = indexed_array_wrapper
            refinement.covered = True
            refinement.comparator = Comparator.LEQ

            if total_accumulated_sum_of_weights < total_sum_of_weights:
                # If the condition separates an example with feature value < 0 from an (sparse) example with feature
                # value == 0
                refinement.threshold = previous_threshold_negative / 2.0
            else:
                # If the condition separates an examples with feature value < 0 from an example with feature value > 0
                refinement.threshold = previous_threshold_negative + (fabs(previous_threshold - previous_threshold_negative) / 2.0)

        # Find and evaluate the best head for the current refinement, if the condition that uses the > operator is
        # used...
        current_head = head_refinement.find_head(best_head, refinement.head, label_indices, refinement_search_ptr.get(),
                                                 True, True)

        if current_head != NULL:
            best_head = current_head
            refinement.head = current_head
            refinement.start = 0
            refinement.end = (last_negative_r + 1)
            refinement.previous = previous_r_negative
            refinement.covered_weights = (total_sum_of_weights - accumulated_sum_of_weights_negative)
            refinement.indexed_array = indexed_array
            refinement.indexed_array_wrapper = indexed_array_wrapper
            refinement.covered = False
            refinement.comparator = Comparator.GR

            if total_accumulated_sum_of_weights < total_sum_of_weights:
                # If the condition separates an example with feature value < 0 from an (sparse) example with feature
                # value == 0
                refinement.threshold = previous_threshold_negative / 2.0
            else:
                # If the condition separates an examples with feature value < 0 from an example with feature value > 0
                refinement.threshold = previous_threshold_negative + (fabs(previous_threshold - previous_threshold_negative) / 2.0)

    return refinement


cdef inline intp __adjust_split(IndexedFloat32Array* indexed_array, intp condition_end, intp condition_previous,
                                float32 threshold):
    """
    Adjusts the position that separates the covered from the uncovered examples with respect to those examples that are
    not contained in the current sub-sample. This requires to look back a certain number of examples, i.e., to traverse
    the examples in ascending or descending order, depending on whether `condition_end` is smaller than
    `condition_previous` or vice versa, until the next example that is contained in the current sub-sample is
    encountered, to see if they satisfy the new condition or not.

    :param indexed_array:       A pointer to a struct of type `IndexedArray` that stores a pointer to a C-array
                                containing the indices of the training examples and the corresponding feature values, as
                                well as the number of elements in said array
    :param condition_end:       The position that separates the covered from the uncovered examples (when only taking
                                into account the examples that are contained in the sample). This is the position to
                                start at
    :param condition_previous:  The position to stop at (exclusive)
    :param threshold:           The threshold of the condition
    :return:                    The adjusted position that separates the covered from the uncovered examples with
                                respect to the examples that are not contained in the sample
    """
    cdef IndexedFloat32* indexed_values = indexed_array.data
    cdef intp adjusted_position = condition_end
    cdef bint ascending = condition_end < condition_previous
    cdef intp direction = 1 if ascending else -1
    cdef intp start = condition_end + direction
    cdef uint32 num_steps = abs(start - condition_previous)
    cdef float32 feature_value
    cdef bint adjust
    cdef uint32 i, r

    # Traverse the examples in ascending (or descending) order until we encounter an example that is contained in the
    # current sub-sample...
    for i in range(num_steps):
        # Check if the current position should be adjusted, or not. This is the case, if the feature value of the
        # current example is smaller than or equal to the given `threshold` (or greater than the `threshold`, if we
        # traverse in descending direction).
        r = start + (i * direction)
        feature_value = indexed_values[r].value
        adjust = (feature_value <= threshold if ascending else feature_value > threshold)

        if adjust:
            # Update the adjusted position and continue...
            adjusted_position = r
        else:
            # If we have found the first example that is separated from the example at the position we started at, we
            # are done...
            break

    return adjusted_position


cdef inline uint32 __filter_current_indices(IndexedFloat32Array* indexed_array,
                                            IndexedFloat32ArrayWrapper* indexed_array_wrapper, intp condition_start,
                                            intp condition_end, Comparator condition_comparator, bint covered,
                                            uint32 num_conditions, uint32[::1] covered_statistics_mask,
                                            uint32 covered_statistics_target, AbstractStatistics* statistics,
                                            uint32[::1] weights):
    """
    Filters an array that contains the indices of the examples that are covered by the previous rule, as well as their
    values for a certain feature, after a new condition that corresponds to said feature has been added, such that the
    filtered array does only contain the indices and feature values of the examples that are covered by the new rule.
    The filtered array is stored in a given struct of type `IndexedFloat32ArrayWrapper` and the given statistics are
    updated accordingly.

    :param indexed_array:               A pointer to a struct of type `IndexedFloat32Array` that stores a pointer to the
                                        C-array to be filtered, as well as the number of elements in said array
    :param indexed_array_wrapper:       A pointer to a struct of type `IndexedFloat32ArrayWrapper` that should be used
                                        to store the filtered array
    :param condition_start:             The element in `indexed_values` that corresponds to the first example
                                        (inclusive) that has been passed to the `RefinementSearch` when searching for
                                        the new condition
    :param condition_end:               The element in `indexed_values` that corresponds to the last example (exclusive)
    :param condition_comparator:        The type of the operator that is used by the new condition
    :param covered                      1, if the examples in range [condition_start, condition_end) are covered by the
                                        new condition and the remaining ones are not, 0, if the examples in said range
                                        are not covered and the remaining ones are
    :param num_conditions:              The total number of conditions in the rule's body (including the new one)
    :param covered_statistics_mask:     An array of type `uint32`, shape `(num_statistics)` that is used to keep track
                                        of the indices of the statistics that are covered by the previous rule. It will
                                        be updated by this function
    :param covered_statistics_target:   The value that is used to mark those elements in `covered_statistics_mask` that
                                        are covered by the previous rule
    :param statistics:                  A pointer to an object of type `AbstractStatistics` to be notified about the
                                        examples that must be considered when searching for the next refinement, i.e.,
                                        the examples that are covered by the new rule
    :param weights:                     An array of type `uint32`, shape `(num_statistics)`, representing the weights of
                                        the training examples
    :return:                            The value that is used to mark those elements in the updated
                                        `covered_statistics_mask` that are covered by the new rule
    """
    cdef IndexedFloat32* indexed_values = indexed_array.data
    cdef uint32 num_indexed_values = indexed_array.numElements
    cdef bint descending = condition_end < condition_start
    cdef uint32 updated_target, weight, index, num_steps, i, r, j
    cdef intp start, end, direction

    # Determine the number of elements in the filtered array...
    cdef uint32 num_condition_steps = abs(condition_start - condition_end)
    cdef uint32 num_elements = num_condition_steps

    if not covered:
        num_elements = (num_indexed_values - num_elements) if num_indexed_values > num_elements else 0

    # Allocate filtered array...
    cdef IndexedFloat32* filtered_array = NULL

    if num_elements > 0:
        filtered_array = <IndexedFloat32*>malloc(num_elements * sizeof(IndexedFloat32))

    if descending:
        direction = -1
        i = num_elements - 1
    else:
        direction = 1
        i = 0

    if covered:
        updated_target = num_conditions
        statistics.resetCoveredStatistics()

        # Retain the indices at positions [condition_start, condition_end) and set the corresponding values in
        # `covered_statistics_mask` to `num_conditions`, which marks them as covered (because
        # `updated_target == num_conditions`)...
        for j in range(num_condition_steps):
            r = condition_start + (j * direction)
            index = indexed_values[r].index
            covered_statistics_mask[index] = num_conditions
            filtered_array[i].index = index
            filtered_array[i].value = indexed_values[r].value
            weight = 1 if weights is None else weights[index]
            statistics.updateCoveredStatistic(index, weight, False)
            i += direction
    else:
        updated_target = covered_statistics_target

        if descending:
            start = num_indexed_values - 1
            end = -1
        else:
            start = 0
            end = num_indexed_values

        if condition_comparator == Comparator.NEQ:
            # Retain the indices at positions [start, condition_start), while leaving the corresponding values in
            # `covered_statistics_mask` untouched, such that all previously covered examples in said range are still
            # marked as covered, while previously uncovered examples are still marked as uncovered...
            num_steps = abs(start - condition_start)

            for j in range(num_steps):
                r = start + (j * direction)
                filtered_array[i].index = indexed_values[r].index
                filtered_array[i].value = indexed_values[r].value
                i += direction

        # Discard the indices at positions [condition_start, condition_end) and set the corresponding values in
        # `covered_statistics_mask` to `num_conditions`, which marks them as uncovered (because
        # `updated_target != num_conditions`)...
        for j in range(num_condition_steps):
            r = condition_start + (j * direction)
            index = indexed_values[r].index
            covered_statistics_mask[index] = num_conditions
            weight = 1 if weights is None else weights[index]
            statistics.updateCoveredStatistic(index, weight, True)

        # Retain the indices at positions [condition_end, end), while leaving the corresponding values in
        # `covered_statistics_mask` untouched, such that all previously covered examples in said range are still marked
        # as covered, while previously uncovered examples are still marked as uncovered...
        num_steps = abs(condition_end - end)

        for j in range(num_steps):
            r = condition_end + (j * direction)
            filtered_array[i].index = indexed_values[r].index
            filtered_array[i].value = indexed_values[r].value
            i += direction

    cdef IndexedFloat32Array* filtered_indexed_array = indexed_array_wrapper.array

    if filtered_indexed_array == NULL:
        filtered_indexed_array = <IndexedFloat32Array*>malloc(sizeof(IndexedFloat32Array))
        indexed_array_wrapper.array = filtered_indexed_array
    else:
        free(filtered_indexed_array.data)

    filtered_indexed_array.data = filtered_array
    filtered_indexed_array.numElements = num_elements
    indexed_array_wrapper.num_conditions = num_conditions
    return updated_target


cdef inline void __filter_any_indices(IndexedFloat32Array* indexed_array,
                                      IndexedFloat32ArrayWrapper* indexed_array_wrapper, uint32 num_conditions,
                                      uint32[::1] covered_statistics_mask, uint32 covered_statistics_target) nogil:
    """
    Filters an array that contains the indices of examples, as well as their values for a certain feature, such that the
    filtered array does only contain the indices and feature values of the examples that are covered by the current
    rule. The filtered array is stored in a given struct of type `IndexedFloat32ArrayWrapper`.

    :param indexed_array:               A pointer to a struct of type `IndexedFloat32Array` that stores a pointer to the
                                        C-array to be filtered, as well as the number of elements in said array
    :param indexed_array_wrapper:       A pointer to a struct of type `IndexedFloat32ArrayWrapper` that should be used
                                        to store the filtered array
    :param num_conditions:              The total number of conditions in the current rule's body
    :param covered_statistics_mask:     An array of type `uint32`, shape `(num_statistics)` that is used to keep track
                                        of the indices of the statistics that are covered by the previous rule. It will
                                        be updated by this function
    :param covered_statistics_target:   The value that is used to mark those elements in `covered_statistics_mask` that
                                        are covered by the previous rule
    """
    cdef IndexedFloat32Array* filtered_indexed_array = indexed_array_wrapper.array
    cdef IndexedFloat32* filtered_array = NULL

    if filtered_indexed_array != NULL:
        filtered_array = filtered_indexed_array.data

    cdef uint32 max_elements = indexed_array.numElements
    cdef uint32 i = 0
    cdef IndexedFloat32* indexed_values
    cdef uint32 index, r

    if max_elements > 0:
        indexed_values = indexed_array.data

        if filtered_array == NULL:
            filtered_array = <IndexedFloat32*>malloc(max_elements * sizeof(IndexedFloat32))

        for r in range(max_elements):
            index = indexed_values[r].index

            if covered_statistics_mask[index] == covered_statistics_target:
                filtered_array[i].index = index
                filtered_array[i].value = indexed_values[r].value
                i += 1

    if i == 0:
        free(filtered_array)
        filtered_array = NULL
    elif i < max_elements:
        filtered_array = <IndexedFloat32*>realloc(filtered_array, i * sizeof(IndexedFloat32))

    if filtered_indexed_array == NULL:
        filtered_indexed_array = <IndexedFloat32Array*>malloc(sizeof(IndexedFloat32Array))

    filtered_indexed_array.data = filtered_array
    filtered_indexed_array.numElements = i
    indexed_array_wrapper.array = filtered_indexed_array
    indexed_array_wrapper.num_conditions = num_conditions


cdef inline Condition __make_condition(uint32 feature_index, Comparator comparator, float32 threshold):
    """
    Creates and returns a new condition.

    :param feature_index:   The index of the feature that is used by the condition
    :param comparator:      The type of the operator used by the condition
    :param threshold:       The threshold that is used by the condition
    """
    cdef Condition condition
    condition.feature_index = feature_index
    condition.comparator = comparator
    condition.threshold = threshold
    return condition


cdef inline void __recalculate_predictions(AbstractStatistics* statistics, uint32 num_statistics,
                                           HeadRefinement head_refinement, uint32[::1] covered_statistics_mask,
                                           uint32 covered_statistics_target, PredictionCandidate* head):
    """
    Updates the scores that a predicted by the head of a rule, based on all available statistics.

    :param statistics:                  A pointer to an object of type `AbstractStatistics` that stores the available
                                        statistics
    :param num_statistics:              The number of available statistics
    :param head_refinement:             The strategy that was used to find the head of the rule
    :param covered_statistics_mask:     An array of type `uint32`, shape `(num_statistics)` that is used to keep track
                                        of the indices of the statistics that are covered by the rule
    :param covered_statistics_target:   The value that is used to mark those elements in `covered_statistics_mask` that
                                        are covered by the rule
    :param head:                        A pointer to an object of type `PredictionCandidate`, representing the head of
                                        the rule
    """
    # The number labels for which the head predicts
    cdef uint32 num_predictions = head.numPredictions_
    # An array that stores the labels for which the head predicts
    cdef uint32* label_indices = head.labelIndices_
    # An array that stores the scores that are predicted by the head
    cdef float64* predicted_scores = head.predictedScores_
    # Temporary variables
    cdef AbstractRefinementSearch* refinement_search
    cdef Prediction* prediction
    cdef float64* updated_scores
    cdef uint32 r, c

    try:
        refinement_search = statistics.beginSearch(num_predictions, label_indices)

        for r in range(num_statistics):
            if covered_statistics_mask[r] == covered_statistics_target:
                refinement_search.updateSearch(r, 1)
                prediction = head_refinement.calculate_prediction(refinement_search, False, False)
                updated_scores = prediction.predictedScores_

                for c in range(num_predictions):
                    predicted_scores[c] = updated_scores[c]
    finally:
        del refinement_search
