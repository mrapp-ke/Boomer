# distutils: language=c++

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement algorithms for inducing individual classification rules.
"""
from boomer.common._arrays cimport uint32, float64, array_uint32, array_intp, get_index
from boomer.common.rules cimport Condition, Comparator
from boomer.common.head_refinement cimport HeadCandidate
from boomer.common.losses cimport Prediction

from libc.math cimport abs
from libc.stdlib cimport qsort

from libcpp.list cimport list as double_linked_list
from libcpp.pair cimport pair

from cython.operator cimport dereference, postincrement

from cpython.mem cimport PyMem_Malloc as malloc, PyMem_Realloc as realloc, PyMem_Free as free


cdef class FeatureMatrix:
    """
    A base class for all classes that provide column-wise access to the feature values of the training examples.
    """

    cdef IndexedArray* get_sorted_feature_values(self, intp feature_index):
        """
        Creates and returns a pointer to a struct of type `IndexedArray` that stores the indices of training examples,
        as well as their feature values, for a specific feature, sorted in ascending order by the feature values.

        :param feature_index:   The index of the feature
        :return:                A pointer to a struct of type `IndexedArray`
        """
        pass


cdef class DenseFeatureMatrix(FeatureMatrix):
    """
    Implements column-wise access to the feature values of the training examples based on a dense feature matrix.

    The feature matrix must be given as a dense Fortran-contiguous array.
    """

    def __cinit__(self, float32[::1, :] x):
        """
        :param x: An array of dtype float, shape `(num_examples, num_features)`, representing the feature values of the
                  training examples
        """
        self.num_examples = x.shape[0]
        self.num_features = x.shape[1]
        self.x = x

    cdef IndexedArray* get_sorted_feature_values(self, intp feature_index):
        # Class members
        cdef float32[::1, :] x = self.x
        # The number of elements to be returned
        cdef intp num_elements = x.shape[0]
        # The array to be returned
        cdef IndexedValue* sorted_array = <IndexedValue*>malloc(num_elements * sizeof(IndexedValue))
        # The struct to be returned
        cdef IndexedArray* indexed_array = <IndexedArray*>malloc(sizeof(IndexedArray))
        dereference(indexed_array).num_elements = num_elements
        dereference(indexed_array).data = sorted_array
        # Temporary variables
        cdef intp i

        for i in range(num_elements):
            sorted_array[i].index = i
            sorted_array[i].value = x[i, feature_index]

        qsort(sorted_array, num_elements, sizeof(IndexedValue), &__compare_indexed_value)
        return indexed_array


cdef class SparseFeatureMatrix(FeatureMatrix):
    """
    Implements column-wise access to the feature values of the training examples based on a sparse feature matrix.

    The feature matrix must be given in compressed sparse column (CSC) format.
    """

    def __cinit__(self, intp num_examples, intp num_features, float32[::1] x_data, intp[::1] x_row_indices,
                  intp[::1] x_col_indices):
        """
        :param num_examples:    The total number of examples
        :param num_features:    The total number of features
        :param x_data:          An array of dtype float, shape `(num_non_zero_feature_values)`, representing the
                                non-zero feature values of the training examples
        :param x_row_indices:   An array of dtype int, shape `(num_non_zero_feature_values)`, representing the
                                row-indices of the examples, the values in `x_data` correspond to
        :param x_col_indices:   An array of dtype int, shape `(num_features + 1)`, representing the indices of the first
                                element in `x_data` and `x_row_indices` that corresponds to a certain feature. The index
                                at the last position is equal to `num_non_zero_feature_values`
        """
        self.num_examples = num_examples
        self.num_features = num_features
        self.x_data = x_data
        self.x_row_indices = x_row_indices
        self.x_col_indices = x_col_indices

    cdef IndexedArray* get_sorted_feature_values(self, intp feature_index):
        # Class members
        cdef float32[::1] x_data = self.x_data
        cdef intp[::1] x_row_indices = self.x_row_indices
        cdef intp[::1] x_col_indices = self.x_col_indices
        # The index of the first element in `x_data` and `x_row_indices` that corresponds to the given feature index
        cdef intp start = x_col_indices[feature_index]
        # The index of the last element in `x_data` and `x_row_indices` that corresponds to the given feature index
        cdef intp end = x_col_indices[feature_index + 1]
        # The number of elements to be returned
        cdef intp num_elements = end - start
        # The struct to be returned
        cdef IndexedArray* indexed_array = <IndexedArray*>malloc(sizeof(IndexedArray))
        dereference(indexed_array).num_elements = num_elements
        # The array to be returned
        cdef IndexedValue* sorted_array = NULL
        # Temporary variables
        cdef intp i, j

        if num_elements > 0:
            sorted_array = <IndexedValue*>malloc(num_elements * sizeof(IndexedValue))
            i = 0

            for j in range(start, end):
                sorted_array[i].index = x_row_indices[j]
                sorted_array[i].value = x_data[j]
                i += 1

            qsort(sorted_array, num_elements, sizeof(IndexedValue), &__compare_indexed_value)

        dereference(indexed_array).data = sorted_array
        return indexed_array


cdef class RuleInduction:
    """
    A base class for all classes that implement an algorithm for the induction of individual classification rules.
    """

    cdef void induce_default_rule(self, uint8[::1, :] y, Loss loss, ModelBuilder model_builder):
        """
        Induces the default rule that minimizes a certain loss function with respect to the given ground truth labels.

        :param y:               An array of dtype float, shape `(num_examples, num_labels)`, representing the ground
                                truth labels of the training examples
        :param loss:            The loss function to be minimized
        :param model_builder:   The builder, the default rule should be added to
        """
        pass

    cdef bint induce_rule(self, intp[::1] nominal_attribute_indices, FeatureMatrix feature_matrix, intp num_labels,
                          HeadRefinement head_refinement, Loss loss, LabelSubSampling label_sub_sampling,
                          InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                          Pruning pruning, Shrinkage shrinkage, intp min_coverage, intp max_conditions,
                          intp max_head_refinements, RNG rng, ModelBuilder model_builder):
        """
        Induces a single- or multi-label classification rule that minimizes a certain loss function for the training
        examples it covers.

        :param nominal_attribute_indices:   An array of dtype int, shape `(num_nominal_attributes)`, representing the
                                            indices of all nominal features (in ascending order) or None, if no nominal
                                            features are available
        :param feature_matrix:              A `FeatureMatrix` that provides column-wise access to the feature values of
                                            the training examples
        :param num_labels:                  The total number of labels
        :param head_refinement:             The strategy that is used to find the heads of rules
        :param loss:                        The loss function to be minimized
        :param label_sub_sampling:          The strategy that should be used to sub-sample the labels or None, if no
                                            label sub-sampling should be used
        :param instance_sub_sampling:       The strategy that should be used to sub-sample the training examples or
                                            None, if no instance sub-sampling should be used
        :param feature_sub_sampling:        The strategy that should be used to sub-sample the available features or
                                            None, if no feature sub-sampling should be used
        :param pruning:                     The strategy that should be used to prune rules or None, if no pruning
                                            should be used
        :param shrinkage:                   The strategy that should be used to shrink the weights of rules or None, if
                                            no shrinkage should be used
        :param min_coverage:                The minimum number of training examples that must be covered by the rule.
                                            Must be at least 1
        :param max_conditions:              The maximum number of conditions to be included in the rule's body. Must be
                                            at least 1 or -1, if the number of conditions should not be restricted
        :param max_head_refinements:        The maximum number of times the head of a rule may be refined after a new 
                                            condition has been added to its body. Must be at least 1 or -1, if the 
                                            number of refinements should not be restricted
        :param rng:                         The random number generator to be used
        :param model_builder:               The builder, the rule should be added to
        :return:                            1, if a rule has been induced, 0 otherwise
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
        self.cache_global = new map[intp, IndexedArray*]()

    def __dealloc__(self):
        cdef map[intp, IndexedArray*]* cache_global = self.cache_global
        cdef map[intp, IndexedArray*].iterator cache_global_iterator = dereference(cache_global).begin()
        cdef IndexedArray* indexed_array

        while cache_global_iterator != dereference(cache_global).end():
            indexed_array = dereference(cache_global_iterator).second
            free(dereference(indexed_array).data)
            free(indexed_array)
            postincrement(cache_global_iterator)

        del self.cache_global

    cdef void induce_default_rule(self, uint8[::1, :] y, Loss loss, ModelBuilder model_builder):
        cdef float64[::1] scores = loss.calculate_default_scores(y)
        model_builder.set_default_rule(scores)

    cdef bint induce_rule(self, intp[::1] nominal_attribute_indices, FeatureMatrix feature_matrix, intp num_labels,
                          HeadRefinement head_refinement, Loss loss, LabelSubSampling label_sub_sampling,
                          InstanceSubSampling instance_sub_sampling, FeatureSubSampling feature_sub_sampling,
                          Pruning pruning, Shrinkage shrinkage, intp min_coverage, intp max_conditions,
                          intp max_head_refinements, RNG rng, ModelBuilder model_builder):
        # The total number of training examples
        cdef intp num_examples = feature_matrix.num_examples
        # The total number of features
        cdef intp num_features = feature_matrix.num_features
        # The head of the induced rule
        cdef HeadCandidate head = None
        # A (stack-allocated) list that contains the conditions in the rule's body (in the order they have been learned)
        cdef double_linked_list[Condition] conditions
        # The total number of conditions
        cdef intp num_conditions = 0
        # An array representing the number of conditions per type of operator
        cdef intp[::1] num_conditions_per_comparator = array_intp(4)
        num_conditions_per_comparator[:] = 0
        # An array that is used to keep track of the indices of the training examples are covered by the current rule.
        # Each element in the array corresponds to the example at the corresponding index. If the value for an element
        # is equal to `covered_examples_target`, it is covered by the current rule, otherwise it is not.
        cdef uint32[::1] covered_examples_mask = array_uint32(num_examples)
        covered_examples_mask[:] = 0
        cdef uint32 covered_examples_target = 0

        # Variables for representing the best refinement
        cdef bint found_refinement = True
        cdef Comparator best_condition_comparator
        cdef intp best_condition_start, best_condition_end, best_condition_previous, best_condition_feature_index
        cdef bint best_condition_covered
        cdef float32 best_condition_threshold
        cdef uint32 best_condition_covered_weights
        cdef IndexedArray* best_condition_indexed_array
        cdef IndexedArrayWrapper* best_condition_indexed_array_wrapper

        # Variables for specifying the examples that should be used for finding the best refinement
        cdef map[intp, IndexedArray*]* cache_global = self.cache_global
        cdef IndexedArray* indexed_array
        cdef map[intp, IndexedArrayWrapper*] cache_local  # Stack-allocated map
        cdef map[intp, IndexedArrayWrapper*].iterator cache_local_iterator
        cdef IndexedArrayWrapper* indexed_array_wrapper
        cdef IndexedValue* indexed_values
        cdef intp num_indexed_values

        # Variables for specifying the features that should be used for finding the best refinement
        cdef intp num_nominal_features = nominal_attribute_indices.shape[0] if nominal_attribute_indices is not None else 0
        cdef intp next_nominal_f = -1
        cdef intp[::1] feature_indices
        cdef intp next_nominal_c, num_sampled_features
        cdef bint nominal

        # Temporary variables
        cdef HeadCandidate current_head
        cdef Prediction prediction
        cdef float64[::1] predicted_scores
        cdef float32 previous_threshold, current_threshold, previous_threshold_negative
        cdef uint32 weight
        cdef intp c, f, r, i, first_r, previous_r, last_negative_r, previous_r_negative

        # Sub-sample examples, if necessary...
        cdef pair[uint32[::1], uint32] uint32_array_scalar_pair
        cdef uint32[::1] weights
        cdef uint32 total_sum_of_weights, sum_of_weights, accumulated_sum_of_weights
        cdef uint32 accumulated_sum_of_weights_negative, total_accumulated_sum_of_weights

        if instance_sub_sampling is None:
            weights = None
            total_sum_of_weights = <uint32>num_examples
        else:
            uint32_array_scalar_pair = instance_sub_sampling.sub_sample(num_examples, rng)
            weights = uint32_array_scalar_pair.first
            total_sum_of_weights = uint32_array_scalar_pair.second

        # Notify the loss function about the examples that are included in the sub-sample...
        loss.begin_instance_sub_sampling()

        for i in range(num_examples):
            weight = 1 if weights is None else weights[i]
            loss.update_sub_sample(i, weight, False)

        # Sub-sample labels, if necessary...
        cdef intp[::1] label_indices

        if label_sub_sampling is None:
            label_indices = None
        else:
            label_indices = label_sub_sampling.sub_sample(num_labels, rng)

        try:
            # Search for the best refinement until no improvement in terms of the rule's quality score is possible
            # anymore or the maximum number of conditions has been reached...
            while found_refinement and (max_conditions == -1 or num_conditions < max_conditions):
                found_refinement = False

                # Sub-sample features, if necessary...
                if feature_sub_sampling is None:
                    feature_indices = None
                    num_sampled_features = num_features
                else:
                    feature_indices = feature_sub_sampling.sub_sample(num_features, rng)
                    num_sampled_features = feature_indices.shape[0]

                # Obtain the index of the first nominal feature, if any...
                if num_nominal_features > 0:
                    next_nominal_f = nominal_attribute_indices[0]
                    next_nominal_c = 1

                # Search for the best condition among all available features to be added to the current rule. For each
                # feature, the examples are traversed in descending order of their respective feature values and the
                # loss function is updated accordingly. For each potential condition, a quality score is calculated to
                # keep track of the best possible refinement.
                for c in range(num_sampled_features):
                    f = get_index(c, feature_indices)

                    # Obtain array that contains the indices of the training examples sorted according to the current
                    # feature...
                    indexed_array_wrapper = cache_local[f]

                    if indexed_array_wrapper == NULL:
                        indexed_array_wrapper = <IndexedArrayWrapper*>malloc(sizeof(IndexedArrayWrapper))
                        dereference(indexed_array_wrapper).array = NULL
                        dereference(indexed_array_wrapper).num_conditions = 0
                        cache_local[f] = indexed_array_wrapper

                    indexed_array = dereference(indexed_array_wrapper).array

                    if indexed_array == NULL:
                        indexed_array = dereference(cache_global)[f]

                        if indexed_array == NULL:
                            indexed_array = feature_matrix.get_sorted_feature_values(f)
                            dereference(cache_global)[f] = indexed_array

                    # Filter indices, if only a subset of the contained examples is covered...
                    if num_conditions > dereference(indexed_array_wrapper).num_conditions:
                        __filter_any_indices(indexed_array, indexed_array_wrapper, num_conditions,
                                             covered_examples_mask, covered_examples_target)
                        indexed_array = dereference(indexed_array_wrapper).array

                    num_indexed_values = dereference(indexed_array).num_elements
                    indexed_values = dereference(indexed_array).data

                    # Check if feature is nominal...
                    if f == next_nominal_f:
                        nominal = True

                        if next_nominal_c < num_nominal_features:
                            next_nominal_f = nominal_attribute_indices[next_nominal_c]
                            next_nominal_c += 1
                        else:
                            next_nominal_f = -1
                    else:
                        nominal = False

                    # Tell the loss function to start a new search when processing a new feature...
                    loss.begin_search(label_indices)

                    # In the following, we start by processing all examples with feature values < 0...
                    sum_of_weights = 0
                    first_r = 0
                    last_negative_r = -1

                    # Traverse examples with feature values < 0 in ascending order until the first example with
                    # weight > 0 is encountered...
                    for r in range(num_indexed_values):
                        current_threshold = indexed_values[r].value

                        if current_threshold >= 0:
                            break

                        last_negative_r = r
                        i = indexed_values[r].index
                        weight = 1 if weights is None else weights[i]

                        if weight > 0:
                            # Tell the loss function that the example will be covered by upcoming refinements...
                            loss.update_search(i, weight)
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
                                    # Find and evaluate the best head for the current refinement, if a condition that
                                    # uses the <= operator (or the == operator in case of a nominal feature) is used...
                                    current_head = head_refinement.find_head(head, label_indices, loss, False, False)

                                    # If the refinement is better than the current rule...
                                    if current_head is not None:
                                        found_refinement = True
                                        head = current_head
                                        best_condition_start = first_r
                                        best_condition_end = r
                                        best_condition_previous = previous_r
                                        best_condition_feature_index = f
                                        best_condition_covered_weights = sum_of_weights
                                        best_condition_indexed_array = indexed_array
                                        best_condition_indexed_array_wrapper = indexed_array_wrapper
                                        best_condition_covered = True

                                        if nominal:
                                            best_condition_comparator = Comparator.EQ
                                            best_condition_threshold = previous_threshold
                                        else:
                                            best_condition_comparator = Comparator.LEQ
                                            best_condition_threshold = (previous_threshold + current_threshold) / 2.0

                                    # Find and evaluate the best head for the current refinement, if a condition that
                                    # uses the > operator (or the != operator in case of a nominal feature) is used...
                                    current_head = head_refinement.find_head(head, label_indices, loss, True, False)

                                    # If the refinement is better than the current rule...
                                    if current_head is not None:
                                        found_refinement = True
                                        head = current_head
                                        best_condition_start = first_r
                                        best_condition_end = r
                                        best_condition_previous = previous_r
                                        best_condition_feature_index = f
                                        best_condition_covered_weights = (total_sum_of_weights - sum_of_weights)
                                        best_condition_indexed_array = indexed_array
                                        best_condition_indexed_array_wrapper = indexed_array_wrapper
                                        best_condition_covered = False

                                        if nominal:
                                            best_condition_comparator = Comparator.NEQ
                                            best_condition_threshold = previous_threshold
                                        else:
                                            best_condition_comparator = Comparator.GR
                                            best_condition_threshold = (previous_threshold + current_threshold) / 2.0

                                    # Reset the loss function in case of a nominal feature, as the previous examples
                                    # will not be covered by the next condition...
                                    if nominal:
                                        loss.reset_search()
                                        sum_of_weights = 0
                                        first_r = r

                                previous_threshold = current_threshold
                                previous_r = r

                                # Tell the loss function that the example will be covered by upcoming refinements...
                                loss.update_search(i, weight)
                                sum_of_weights += weight
                                accumulated_sum_of_weights += weight

                        # If the feature is nominal and the examples that have been iterated so far do not all have the
                        # same feature value, or if not all examples have been iterated so far, we must evaluate
                        # additional conditions `f == previous_threshold` and `f != previous_threshold`...
                        if nominal and sum_of_weights > 0 and (sum_of_weights < accumulated_sum_of_weights
                                                               or accumulated_sum_of_weights < total_sum_of_weights):
                            # Find and evaluate the best head for the current refinement, if a condition that uses the
                            # == operator is used...
                            current_head = head_refinement.find_head(head, label_indices, loss, False, False)

                            # If the refinement is better than the current rule...
                            if current_head is not None:
                                found_refinement = True
                                head = current_head
                                best_condition_start = first_r
                                best_condition_end = (last_negative_r + 1)
                                best_condition_previous = previous_r
                                best_condition_feature_index = f
                                best_condition_covered_weights = sum_of_weights
                                best_condition_indexed_array = indexed_array
                                best_condition_indexed_array_wrapper = indexed_array_wrapper
                                best_condition_covered = True
                                best_condition_comparator = Comparator.EQ
                                best_condition_threshold = previous_threshold

                            # Find and evaluate the best head for the current refinement, if a condition that uses the !=
                            # operator is used...
                            current_head = head_refinement.find_head(head, label_indices, loss, True, False)

                            # If the refinement is better than the current rule...
                            if current_head is not None:
                                found_refinement = True
                                head = current_head
                                best_condition_start = first_r
                                best_condition_end = (last_negative_r + 1)
                                best_condition_previous = previous_r
                                best_condition_feature_index = f
                                best_condition_covered_weights = (total_sum_of_weights - sum_of_weights)
                                best_condition_indexed_array = indexed_array
                                best_condition_indexed_array_wrapper = indexed_array_wrapper
                                best_condition_covered = False
                                best_condition_comparator = Comparator.NEQ
                                best_condition_threshold = previous_threshold

                        # Reset the loss function, if any examples with feature value < 0 have been processed...
                        loss.reset_search()

                    previous_threshold_negative = previous_threshold
                    previous_r_negative = previous_r
                    accumulated_sum_of_weights_negative = accumulated_sum_of_weights

                    # We continue by processing all examples with feature values >= 0...
                    sum_of_weights = 0
                    first_r = num_indexed_values - 1

                    # Traverse examples with feature values >= 0 in descending order until the first example with
                    # weight > 0 is encountered...
                    for r in range(first_r, last_negative_r, -1):
                        i = indexed_values[r].index
                        weight = 1 if weights is None else weights[i]

                        if weight > 0:
                            # Tell the loss function that the example will be covered by upcoming refinements...
                            loss.update_search(i, weight)
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
                                    # Find and evaluate the best head for the current refinement, if a condition that
                                    # uses the > operator (or the == operator in case of a nominal feature) is used...
                                    current_head = head_refinement.find_head(head, label_indices, loss, False, False)

                                    # If the refinement is better than the current rule...
                                    if current_head is not None:
                                        found_refinement = True
                                        head = current_head
                                        best_condition_start = first_r
                                        best_condition_end = r
                                        best_condition_previous = previous_r
                                        best_condition_feature_index = f
                                        best_condition_covered_weights = sum_of_weights
                                        best_condition_indexed_array = indexed_array
                                        best_condition_indexed_array_wrapper = indexed_array_wrapper
                                        best_condition_covered = True

                                        if nominal:
                                            best_condition_comparator = Comparator.EQ
                                            best_condition_threshold = previous_threshold
                                        else:
                                            best_condition_comparator = Comparator.GR
                                            best_condition_threshold = (previous_threshold + current_threshold) / 2.0

                                    # Find and evaluate the best head for the current refinement, if a condition that
                                    # uses the <= operator (or the != operator in case of a nominal feature) is used...
                                    current_head = head_refinement.find_head(head, label_indices, loss, True, False)

                                    # If the refinement is better than the current rule...
                                    if current_head is not None:
                                        found_refinement = True
                                        head = current_head
                                        best_condition_start = first_r
                                        best_condition_end = r
                                        best_condition_previous = previous_r
                                        best_condition_feature_index = f
                                        best_condition_covered_weights = (total_sum_of_weights - sum_of_weights)
                                        best_condition_indexed_array = indexed_array
                                        best_condition_indexed_array_wrapper = indexed_array_wrapper
                                        best_condition_covered = False

                                        if nominal:
                                            best_condition_comparator = Comparator.NEQ
                                            best_condition_threshold = previous_threshold
                                        else:
                                            best_condition_comparator = Comparator.LEQ
                                            best_condition_threshold = (previous_threshold + current_threshold) / 2.0

                                    # Reset the loss function in case of a nominal feature, as the previous examples
                                    # will not be covered by the next condition...
                                    if nominal:
                                        loss.reset_search()
                                        sum_of_weights = 0
                                        first_r = r

                                previous_threshold = current_threshold
                                previous_r = r

                                # Tell the loss function that the example will be covered by upcoming refinements...
                                loss.update_search(i, weight)
                                sum_of_weights += weight
                                accumulated_sum_of_weights += weight

                    # If the feature is nominal and the examples with feature values >= 0 that have been iterated so far
                    # do not all have the same feature value, we must evaluate additional conditions
                    # `f == previous_threshold` and `f != previous_threshold`...
                    if nominal and sum_of_weights > 0 and sum_of_weights < accumulated_sum_of_weights:
                        # Find and evaluate the best head for the current refinement, if a condition that uses the ==
                        # operator is used...
                        current_head = head_refinement.find_head(head, label_indices, loss, False, False)

                        # If the refinement is better than the current rule...
                        if current_head is not None:
                            found_refinement = True
                            head = current_head
                            best_condition_start = first_r
                            best_condition_end = last_negative_r
                            best_condition_previous = previous_r
                            best_condition_feature_index = f
                            best_condition_covered_weights = sum_of_weights
                            best_condition_indexed_array = indexed_array
                            best_condition_indexed_array_wrapper = indexed_array_wrapper
                            best_condition_covered = True
                            best_condition_comparator = Comparator.EQ
                            best_condition_threshold = previous_threshold

                        # Find and evaluate the best head for the current refinement, if a condition that uses the !=
                        # operator is used...
                        current_head = head_refinement.find_head(head, label_indices, loss, True, False)

                        # If the refinement is better than the current rule...
                        if current_head is not None:
                            found_refinement = True
                            head = current_head
                            best_condition_start = first_r
                            best_condition_end = last_negative_r
                            best_condition_previous = previous_r
                            best_condition_feature_index = f
                            best_condition_covered_weights = (total_sum_of_weights - sum_of_weights)
                            best_condition_indexed_array = indexed_array
                            best_condition_indexed_array_wrapper = indexed_array_wrapper
                            best_condition_covered = False
                            best_condition_comparator = Comparator.NEQ
                            best_condition_threshold = previous_threshold

                    total_accumulated_sum_of_weights = accumulated_sum_of_weights_negative + accumulated_sum_of_weights

                    # If the sum of weights of all examples that have been iterated so far (including those with feature
                    # values < 0 and those with feature values >= 0) is less than the sum of of weights of all examples,
                    # this means that there are examples with sparse, i.e. zero, feature values. In such case, we must
                    # explicitly test conditions that separate these examples from the ones that have already been
                    # iterated...
                    if total_accumulated_sum_of_weights > 0 and total_accumulated_sum_of_weights < total_sum_of_weights:
                        # If the feature is nominal, we must reset the loss function once again to ensure that the
                        # accumulated state includes all examples that have been processed so far...
                        if nominal:
                            loss.reset_search()
                            first_r = num_indexed_values - 1

                        # Find and evaluate the best head for the current refinement, if the condition
                        # `f > previous_threshold / 2` (or the condition `f != 0` in case of a nominal feature) is
                        # used...
                        current_head = head_refinement.find_head(head, label_indices, loss, False, nominal)

                        # If the refinement is better than the current rule...
                        if current_head is not None:
                            found_refinement = True
                            head = current_head
                            best_condition_start = first_r
                            best_condition_feature_index = f
                            best_condition_indexed_array = indexed_array
                            best_condition_indexed_array_wrapper = indexed_array_wrapper
                            best_condition_covered = True

                            if nominal:
                                best_condition_end = -1
                                best_condition_previous = -1
                                best_condition_covered_weights = total_accumulated_sum_of_weights
                                best_condition_comparator = Comparator.NEQ
                                best_condition_threshold = 0.0
                            else:
                                best_condition_end = last_negative_r
                                best_condition_previous = previous_r
                                best_condition_covered_weights = accumulated_sum_of_weights
                                best_condition_comparator = Comparator.GR
                                best_condition_threshold = previous_threshold / 2.0

                        # Find and evaluate the best head for the current refinement, if the condition
                        # `f <= previous_threshold / 2` (or `f == 0` in case of a nominal feature) is used...
                        current_head = head_refinement.find_head(head, label_indices, loss, True, nominal)

                        # If the refinement is better than the current rule...
                        if current_head is not None:
                            found_refinement = True
                            head = current_head
                            best_condition_start = first_r
                            best_condition_feature_index = f
                            best_condition_indexed_array = indexed_array
                            best_condition_indexed_array_wrapper = indexed_array_wrapper
                            best_condition_covered = False

                            if nominal:
                                best_condition_end = -1
                                best_condition_previous = -1
                                best_condition_covered_weights = (total_sum_of_weights - total_accumulated_sum_of_weights)
                                best_condition_comparator = Comparator.EQ
                                best_condition_threshold = 0.0
                            else:
                                best_condition_end = last_negative_r
                                best_condition_previous = previous_r
                                best_condition_covered_weights = (total_sum_of_weights - accumulated_sum_of_weights)
                                best_condition_comparator = Comparator.LEQ
                                best_condition_threshold = previous_threshold / 2.0

                    # If the feature is numerical and there are other examples than those with feature values < 0 that
                    # have been processed earlier, we must evaluate additional conditions that separate the examples
                    # with feature values < 0 from the remaining ones (unlike in the nominal case, these conditions
                    # cannot be evaluated earlier, because it remains unclear what the thresholds of the conditions
                    # should be until the examples with feature values >= 0 have been processed).
                    if not nominal and accumulated_sum_of_weights_negative > 0 and accumulated_sum_of_weights_negative < total_sum_of_weights:
                        # Find and evaluate the best head for the current refinement, if the condition that uses the <=
                        # operator is used...
                        current_head = head_refinement.find_head(head, label_indices, loss, False, True)

                        if current_head is not None:
                            found_refinement = True
                            head = current_head
                            best_condition_start = 0
                            best_condition_end = (last_negative_r + 1)
                            best_condition_previous = previous_r_negative
                            best_condition_feature_index = f
                            best_condition_covered_weights = accumulated_sum_of_weights_negative
                            best_condition_indexed_array = indexed_array
                            best_condition_indexed_array_wrapper = indexed_array_wrapper
                            best_condition_covered = True
                            best_condition_comparator = Comparator.LEQ

                            if total_accumulated_sum_of_weights < total_sum_of_weights:
                                # If the condition separates an example with feature value < 0 from an (sparse) example
                                # with feature value == 0
                                best_condition_threshold = previous_threshold_negative / 2.0
                            else:
                                # If the condition separates an examples with feature value < 0 from an example with
                                # feature value > 0
                                best_condition_threshold = previous_threshold_negative + (abs(previous_threshold - previous_threshold_negative) / 2.0)

                        # Find and evaluate the best head for the current refinement, if the condition that uses the >
                        # operator is used...
                        current_head = head_refinement.find_head(head, label_indices, loss, True, True)

                        if current_head is not None:
                            found_refinement = True
                            head = current_head
                            best_condition_start = 0
                            best_condition_end = (last_negative_r + 1)
                            best_condition_previous = previous_r_negative
                            best_condition_feature_index = f
                            best_condition_covered_weights = (total_sum_of_weights - accumulated_sum_of_weights_negative)
                            best_condition_indexed_array = indexed_array
                            best_condition_indexed_array_wrapper = indexed_array_wrapper
                            best_condition_covered = False
                            best_condition_comparator = Comparator.GR

                            if total_accumulated_sum_of_weights < total_sum_of_weights:
                                # If the condition separates an example with feature value < 0 from an (sparse) example
                                # with feature value == 0
                                best_condition_threshold = previous_threshold_negative / 2.0
                            else:
                                # If the condition separates an examples with feature value < 0 from an example with
                                # feature value > 0
                                best_condition_threshold = previous_threshold_negative + (abs(previous_threshold - previous_threshold_negative) / 2.0)

                if found_refinement:
                    # If a refinement has been found, add the new condition...
                    conditions.push_back(__make_condition(best_condition_feature_index, best_condition_comparator,
                                                          best_condition_threshold))
                    num_conditions += 1
                    num_conditions_per_comparator[<intp>best_condition_comparator] += 1

                    if max_head_refinements > 0 and num_conditions >= max_head_refinements:
                        # Keep the labels for which the rule predicts, if the head should not be further refined...
                        label_indices = head.label_indices

                    # If instance sub-sampling is used, examples that are not contained in the current sub-sample were
                    # not considered for finding the new condition. In the next step, we need to identify the examples
                    # that are covered by the refined rule, including those that are not contained in the sub-sample,
                    # via the function `__filter_current_indices`. Said function calculates the number of covered
                    # examples based on the variable `best_condition_end`, which represents the position that separates
                    # the covered from the uncovered examples. However, when taking into account the examples that are
                    # not contained in the sub-sample, this position may differ from the current value of
                    # `best_condition_end` and therefore must be adjusted...
                    if weights is not None and abs(best_condition_previous - best_condition_end) > 1:
                        best_condition_end = __adjust_split(best_condition_indexed_array, best_condition_end,
                                                            best_condition_previous, best_condition_threshold)

                    # Identify the examples for which the rule predicts...
                    covered_examples_target = __filter_current_indices(best_condition_indexed_array,
                                                                       best_condition_indexed_array_wrapper,
                                                                       best_condition_start, best_condition_end,
                                                                       best_condition_comparator,
                                                                       best_condition_covered, num_conditions,
                                                                       covered_examples_mask, covered_examples_target,
                                                                       loss, weights)
                    total_sum_of_weights = best_condition_covered_weights

                    if total_sum_of_weights <= min_coverage:
                        # Abort refinement process if the rule is not allowed to cover less examples...
                        break

            if head is None:
                # No rule could be induced, because no useful condition could be found. This is for example the case, if
                # all features are constant.
                return False
            else:
                label_indices = head.label_indices
                predicted_scores = head.predicted_scores

                if weights is not None:
                    # Prune rule, if necessary (a rule can only be pruned if it contains more than one condition)...
                    if pruning is not None and num_conditions > 1:
                        uint32_array_scalar_pair = pruning.prune(cache_global, conditions, covered_examples_mask,
                                                                 covered_examples_target, weights, label_indices, loss,
                                                                 head_refinement)
                        covered_examples_mask = uint32_array_scalar_pair.first
                        covered_examples_target = uint32_array_scalar_pair.second

                    # If instance sub-sampling is used, we need to re-calculate the scores in the head based on the
                    # entire training data...
                    loss.begin_search(label_indices)

                    for r in range(num_examples):
                        if covered_examples_mask[r] == covered_examples_target:
                            loss.update_search(r, 1)

                    prediction = head_refinement.evaluate_predictions(loss, False, False)
                    predicted_scores[:] = prediction.predicted_scores

                # Apply shrinkage, if necessary...
                if shrinkage is not None:
                    shrinkage.apply_shrinkage(predicted_scores)

                # Tell the loss function that a new rule has been induced...
                for r in range(num_examples):
                    if covered_examples_mask[r] == covered_examples_target:
                        loss.apply_prediction(r, label_indices, predicted_scores)

                # Add the induced rule to the model...
                model_builder.add_rule(label_indices, predicted_scores, conditions, num_conditions_per_comparator)
                return True
        finally:
            # Free memory occupied by the arrays stored in `cache_local`...
            cache_local_iterator = cache_local.begin()

            while cache_local_iterator != cache_local.end():
                indexed_array_wrapper = dereference(cache_local_iterator).second
                indexed_array = dereference(indexed_array_wrapper).array

                if indexed_array != NULL:
                    indexed_values = dereference(indexed_array).data
                    free(indexed_values)

                free(indexed_array)
                free(indexed_array_wrapper)
                postincrement(cache_local_iterator)


cdef int __compare_indexed_value(const void* a, const void* b) nogil:
    """
    Compares the values of two structs of type `IndexedValue`.

    :param a:   A pointer to the first struct
    :param b:   A pointer to the second struct
    :return:    -1 if the value of the first struct is smaller than the value of the second struct, 0 if both values are
                equal, or 1 if the value of the first struct is greater than the value of the second struct
    """
    cdef float32 v1 = (<IndexedValue*>a).value
    cdef float32 v2 = (<IndexedValue*>b).value
    return -1 if v1 < v2 else (0 if v1 == v2 else 1)


cdef inline Condition __make_condition(intp feature_index, Comparator comparator, float32 threshold):
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


cdef inline intp __adjust_split(IndexedArray* indexed_array, intp condition_end, intp condition_previous,
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
    cdef IndexedValue* indexed_values = dereference(indexed_array).data
    cdef intp adjusted_position = condition_end
    cdef bint ascending = condition_end < condition_previous
    cdef intp direction = 1 if ascending else -1
    cdef float32 feature_value
    cdef bint adjust
    cdef intp r

    # Traverse the examples in ascending (or descending) order until we encounter an example that is contained in the
    # current sub-sample...
    for r in range(condition_end + direction, condition_previous, direction):
        # Check if the current position should be adjusted, or not. This is the case, if the feature value of the
        # current example is smaller than or equal to the given `threshold` (or greater than the `threshold`, if we
        # traverse in descending direction).
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


cdef inline uint32 __filter_current_indices(IndexedArray* indexed_array, IndexedArrayWrapper* indexed_array_wrapper,
                                            intp condition_start, intp condition_end, Comparator condition_comparator,
                                            bint covered, intp num_conditions, uint32[::1] covered_examples_mask,
                                            uint32 covered_examples_target, Loss loss, uint32[::1] weights):
    """
    Filters an array that contains the indices of the examples that are covered by the previous rule, as well as their
    values for a certain feature, after a new condition that corresponds to said feature has been added, such that the
    filtered array does only contain the indices and feature values of the examples that are covered by the new rule.
    The filtered array is stored in a given struct of type `IndexedArrayWrapper` and the given loss function is updated
    accordingly.

    :param indexed_array:           A pointer to a struct of type `IndexedArray` that stores a pointer to the C-array to
                                    be filtered, as well as the number of elements in said array
    :param indexed_array_wrapper:   A pointer to a struct of type `IndexedArrayWrapper` that should be used to store the
                                    filtered array
    :param condition_start:         The element in `indexed_values` that corresponds to the first example (inclusive)
                                    that has been passed to the loss function when searching for the new condition
    :param condition_end:           The element in `indexed_values` that corresponds to the last example (exclusive)
    :param condition_comparator:    The type of the operator that is used by the new condition
    :param covered                  1, if the examples in range [condition_start, condition_end) are covered by the new
                                    condition and the remaining ones are not, 0, if the examples in said range are not
                                    covered and the remaining ones are
    :param num_conditions:          The total number of conditions in the rule's body (including the new one)
    :param covered_examples_mask:   An array of dtype uint, shape `(num_examples)` that is used to keep track of the
                                    indices of the examples that are covered by the previous rule. It will be updated by
                                    this function
    :param covered_examples_target: The value that is used to mark those elements in `covered_examples_mask` that are
                                    covered by the previous rule
    :param loss:                    The loss function to be notified about the examples that must be considered when
                                    searching for the next refinement, i.e., the examples that are covered by the new
                                    rule
    :param weights:                 An array of dtype uint, shape `(num_examples)`, representing the weights of the
                                    training examples
    :return:                        The value that is used to mark those elements in the updated `covered_examples_mask`
                                    that are covered by the new rule
    """
    cdef IndexedValue* indexed_values = dereference(indexed_array).data
    cdef intp num_indexed_values = dereference(indexed_array).num_elements
    cdef bint descending = condition_end < condition_start
    cdef uint32 updated_target, weight
    cdef intp start, end, direction, i, r, index

    # Determine the number of elements in the filtered array...
    cdef intp num_elements = abs(condition_start - condition_end)

    if not covered:
        num_elements = num_indexed_values - num_elements

    # Allocate filtered array...
    cdef IndexedValue* filtered_array = NULL

    if num_elements > 0:
        filtered_array = <IndexedValue*>malloc(num_elements * sizeof(IndexedValue))

    if descending:
        direction = -1
        i = num_elements - 1
    else:
        direction = 1
        i = 0

    if covered:
        updated_target = num_conditions
        loss.begin_instance_sub_sampling()

        # Retain the indices at positions [condition_start, condition_end) and set the corresponding values in
        # `covered_examples_mask` to `num_conditions`, which marks them as covered (because
        # `updated_target == num_conditions`)...
        for r in range(condition_start, condition_end, direction):
            index = indexed_values[r].index
            covered_examples_mask[index] = num_conditions
            filtered_array[i].index = index
            filtered_array[i].value = indexed_values[r].value
            weight = 1 if weights is None else weights[index]
            loss.update_sub_sample(index, weight, False)
            i += direction
    else:
        updated_target = covered_examples_target

        if descending:
            start = num_indexed_values - 1
            end = -1
        else:
            start = 0
            end = num_indexed_values

        if condition_comparator == Comparator.NEQ:
            # Retain the indices at positions [start, condition_start), while leaving the corresponding values in
            # `covered_examples_mask` untouched, such that all previously covered examples in said range are still
            # marked as covered, while previously uncovered examples are still marked as uncovered...
            for r in range(start, condition_start, direction):
                filtered_array[i].index = indexed_values[r].index
                filtered_array[i].value = indexed_values[r].value
                i += direction

        # Discard the indices at positions [condition_start, condition_end) and set the corresponding values in
        # `covered_examples_mask` to `num_conditions`, which marks them as uncovered (because
        # `updated_target != num_conditions`)...
        for r in range(condition_start, condition_end, direction):
            index = indexed_values[r].index
            covered_examples_mask[index] = num_conditions
            weight = 1 if weights is None else weights[index]
            loss.update_sub_sample(index, weight, True)

        # Retain the indices at positions [condition_end, end), while leaving the corresponding values in
        # `covered_examples_mask` untouched, such that all previously covered examples in said range are still marked as
        # covered, while previously uncovered examples are still marked as uncovered...
        for r in range(condition_end, end, direction):
            filtered_array[i].index = indexed_values[r].index
            filtered_array[i].value = indexed_values[r].value
            i += direction

    cdef IndexedArray* filtered_indexed_array = dereference(indexed_array_wrapper).array

    if filtered_indexed_array == NULL:
        filtered_indexed_array = <IndexedArray*>malloc(sizeof(IndexedArray))
        dereference(indexed_array_wrapper).array = filtered_indexed_array
    else:
        free(dereference(filtered_indexed_array).data)

    dereference(filtered_indexed_array).data = filtered_array
    dereference(filtered_indexed_array).num_elements = num_elements
    dereference(indexed_array_wrapper).num_conditions = num_conditions
    return updated_target


cdef inline void __filter_any_indices(IndexedArray* indexed_array, IndexedArrayWrapper* indexed_array_wrapper,
                                      intp num_conditions, uint32[::1] covered_examples_mask,
                                      uint32 covered_examples_target):
    """
    Filters an array that contains the indices of examples, as well as their values for a certain feature, such that the
    filtered array does only contain the indices and feature values of the examples that are covered by the current
    rule. The filtered array is stored in a given struct of type `IndexedArrayWrapper`.

    :param indexed_array:           A pointer to a struct of type `IndexedArray` that stores a pointer to the C-array to
                                    be filtered, as well as the number of elements in said array
    :param indexed_array_wrapper:   A pointer to a struct of type `IndexedArrayWrapper` that should be used to store the
                                    filtered array
    :param num_conditions:          The total number of conditions in the current rule's body
    :param covered_examples_mask:   An array of dtype uint, shape `(num_examples)` that is used to keep track of the
                                    indices of the examples that are covered by the previous rule. It will be updated by
                                    this function
    :param covered_examples_target: The value that is used to mark those elements in `covered_examples_mask` that are
                                    covered by the previous rule
    """
    cdef IndexedArray* filtered_indexed_array = dereference(indexed_array_wrapper).array
    cdef IndexedValue* filtered_array = NULL

    if filtered_indexed_array != NULL:
        filtered_array = dereference(filtered_indexed_array).data

    cdef intp max_elements = dereference(indexed_array).num_elements
    cdef intp i = 0
    cdef IndexedValue* indexed_values
    cdef intp r, index

    if max_elements > 0:
        indexed_values = dereference(indexed_array).data

        if filtered_array == NULL:
            filtered_array = <IndexedValue*>malloc(max_elements * sizeof(IndexedValue))

        for r in range(max_elements):
            index = indexed_values[r].index

            if covered_examples_mask[index] == covered_examples_target:
                filtered_array[i].index = index
                filtered_array[i].value = indexed_values[r].value
                i += 1

    if i == 0:
        free(filtered_array)
        filtered_array = NULL
    elif i < max_elements:
        filtered_array = <IndexedValue*>realloc(filtered_array, i * sizeof(IndexedValue))

    if filtered_indexed_array == NULL:
        filtered_indexed_array = <IndexedArray*>malloc(sizeof(IndexedArray))

    dereference(filtered_indexed_array).data = filtered_array
    dereference(filtered_indexed_array).num_elements = i
    dereference(indexed_array_wrapper).array = filtered_indexed_array
    dereference(indexed_array_wrapper).num_conditions = num_conditions
