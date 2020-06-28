# distutils: language=c++

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for sub-sampling training examples, features or labels.
"""
from boomer.common._arrays cimport float64, array_uint32, array_intp

from libc.math cimport log2

from libcpp.unordered_set cimport unordered_set as set


cdef class InstanceSubSampling:
    """
    A base class for all classes that implement a strategy for sub-sampling training examples.
    """

    cdef pair[uint32[::1], uint32] sub_sample(self, intp num_examples, RNG rng):
        """
        Creates and returns a sub-sample of the available training examples.

        :param num_examples:    The total number of available training examples
        :param rng:             The random number generator to be used
        :return:                A pair that contains an array of dtype uint, shape `(num_examples)`, representing the
                                weights of the given training examples, i.e., how many times each of the examples is
                                contained in the sample, as well as the sum of the weights
        """
        pass


cdef class Bagging(InstanceSubSampling):
    """
    Implements bootstrap aggregating (bagging) for drawing a subset (of predefined size) from the available training
    examples with replacement.
    """

    def __cinit__(self, float32 sample_size = 1.0):
        """
        :param sample_size: The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
                            60 % of the available examples). Must be in (0, 1]
        """
        self.sample_size = sample_size

    cdef pair[uint32[::1], uint32] sub_sample(self, intp num_examples, RNG rng):
        cdef float32 sample_size = self.sample_size
        cdef intp num_samples = <intp>(sample_size * num_examples)
        cdef uint32[::1] weights = array_uint32(num_examples)
        cdef uint32 random_index
        cdef intp i

        weights[:] = 0

        for i in range(num_samples):
            # Randomly select the index of an example...
            random_index = rng.random(0, num_examples)

            # Update weight at the selected index...
            weights[random_index] += 1

        cdef pair[uint32[::1], uint32] result  # Stack-allocated pair
        result.first = weights
        result.second = <uint32>num_samples
        return result


cdef class RandomInstanceSubsetSelection(InstanceSubSampling):
    """
    Implements random instance subset selection for drawing a subset (of predefined size) from the available training
    examples without replacement.
    """

    def __cinit__(self, float32 sample_size = 0.66):
        """
        param sample_size: The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
                           60 % of the available examples). Must be in (0, 1)
        """
        self.sample_size = sample_size

    cdef pair[uint32[::1], uint32] sub_sample(self, intp num_examples, RNG rng):
        cdef float32 sample_size = self.sample_size
        cdef intp num_samples = <intp>(sample_size * num_examples)
        cdef uint32[::1] weights = __sample_weights_without_replacement(num_examples, num_samples, rng)
        cdef pair[uint32[::1], uint32] result  # Stack-allocated pair
        result.first = weights
        result.second = <uint32>num_samples
        return result


cdef class FeatureSubSampling:
    """
    A base class for all classes that implement a strategy for sub-sampling features.
    """

    cdef intp[::1] sub_sample(self, intp num_features, RNG rng):
        """
        Creates and returns a sub-sample of the available features.

        :param num_features:    The total number of available features
        :param rng:             The random number generator to be used
        :return:                An array of dtype int, shape `(num_samples)`, representing the indices of the features
                                contained in the sub-sample
        """
        pass

cdef class RandomFeatureSubsetSelection(FeatureSubSampling):
    """
    Implements random feature subset selection for selecting a random subset (of predefined size) from the available
    features.
    """

    def __cinit__(self, float32 sample_size = 0.0):
        """
        :param sample_size: The fraction of features to be included in the sample (e.g. a value of 0.6 corresponds to
                            60 % of the available features). Must be in (0, 1) or 0, if the default sample size
                            `floor(log2(num_features - 1) + 1)` should be used
        """
        self.sample_size = sample_size

    cdef intp[::1] sub_sample(self, intp num_features, RNG rng):
         cdef float32 sample_size = self.sample_size
         cdef intp num_samples

         if sample_size > 0:
            num_samples = <intp>(sample_size * num_features)
         else:
            num_samples = <intp>(log2(num_features - 1) + 1)

         return __sample_indices_without_replacement(num_features, num_samples, rng)


cdef class LabelSubSampling:
    """
    A base class for all classes that implement a strategy for sub-sampling labels.
    """

    cdef intp[::1] sub_sample(self, intp num_labels, RNG rng):
        """
        Creates and returns a sub-sample of the available labels.
        
        :param num_labels:  The total number of available labels
        :param rng:         The random number generator to be used
        :return:            An array of dtype int, shape `(num_samples)`, representing the indices of the labels
                            contained in the sub-sample
        """
        pass


cdef class RandomLabelSubsetSelection(LabelSubSampling):

    def __cinit__(self, intp num_samples):
        """
        :param num_samples: The number of labels to be included in the sample
        """
        self.num_samples = num_samples

    cdef intp[::1] sub_sample(self, intp num_labels, RNG rng):
        cdef intp num_samples = self.num_samples
        return __sample_indices_without_replacement(num_labels, num_samples, rng)


cdef inline uint32[::1] __sample_weights_without_replacement(intp num_total, intp num_samples, RNG rng):
    """
    Randomly selects `num_samples` out of `num_total` elements and sets their weights to 1, while the remaining weights
    are set to 0. The method that is used internally is chosen automatically, depending on the ratio
    `num_samples / num_total`.

    :param num_total:   The total number of available elements
    :param num_samples: The number of weights to be set to 1
    :param rng:         The random number generator to be used
    :return:            An array of dtype uint, shape `(num_total)`, representing the weights of the elements
    """
    cdef float64 ratio = (<float64>num_samples) / (<float64>num_total) if num_total != 0 else 1.0

    if ratio < 0.06:
        # For very small ratios use tracking selection
        return __sample_weights_without_replacement_via_tracking_selection(num_total, num_samples, rng)
    else:
        # Otherwise, use a pool as the default method
        return __sample_weights_without_replacement_via_pool(num_total, num_samples, rng)


cdef inline uint32[::1] __sample_weights_without_replacement_via_tracking_selection(intp num_total, intp num_samples,
                                                                                    RNG rng):
    """
    Randomly selects `num_samples` out of `num_total` elements and sets their weights to 1, while the remaining weights
    are set to 0, by using a set to keep track of the elements that have already been selected. This method is suitable
    if `num_samples` is much smaller than `num_total`.

    :param num_total:   The total number of available elements
    :param num_samples: The number of weights to be set to 1
    :param rng:         The random number generator to be used
    :return:            An array of dtype uint, shape `(num_total)`, representing the weights of the elements
    """
    cdef uint32[::1] weights = array_uint32(num_total)
    cdef set[uint32] selected_indices  # Stack-allocated set
    cdef bint should_continue
    cdef uint32 random_index
    cdef intp i

    weights[:] = 0

    for i in range(num_samples):
        should_continue = True

        while should_continue:
            random_index = rng.random(0, num_total)
            should_continue = not selected_indices.insert(random_index).second

        weights[random_index] = 1

    return weights


cdef inline uint32[::1] __sample_weights_without_replacement_via_pool(intp num_total, intp num_samples, RNG rng):
    """
    Randomly selects `num_samples` out of `num_total` elements and sets their weights to 1, while the remaining weights
    are set to 0, by using a pool, i.e., an array, to keep track of the elements that have not been selected yet.

    :param num_total:   The total number of available elements
    :param num_samples: The number of weights to be set to 1
    :param rng:         The random number generator to be used
    :return:            An array of dtype uint, shape `(num_total)`, representing the weights of the elements
    """
    cdef uint32[::1] weights = array_uint32(num_total)
    cdef intp[::1] pool = array_intp(num_total)
    cdef uint32 random_index, j
    cdef intp i

    # Initialize arrays...
    for i in range(num_total):
        weights[i] = 0
        pool[i] = i

    for i in range(num_samples):
        # Randomly select an index that has not been drawn yet...
        random_index = rng.random(0, num_total - i)
        j = pool[random_index]

        # Set weight at the selected index to 1...
        weights[j] = 1

        # Move the index at the border to the position of the recently drawn index...
        pool[random_index] = pool[num_total - i - 1]

    return weights


cdef inline intp[::1] __sample_indices_without_replacement(intp num_total, intp num_samples, RNG rng):
    """
    Randomly selects `num_samples` out of `num_total` indices without replacement. The method that is used internally is
    chosen automatically, depending on the ratio `num_samples / num_total`.

    :param num_total:   The total number of available indices
    :param num_samples: The number of indices to be sampled
    :param rng:         The random number generator to be used
    :return:            An array of dtype int, shape `(num_samples)`, representing the indices contained in the
                        sub-sample
    """
    cdef float64 ratio = (<float64>num_samples) / (<float64>num_total) if num_total != 0 else 1.0

    # The thresholds for choosing a suitable method are based on empirical experiments
    if ratio < 0.06:
        # For very small ratios use tracking selection
        return __sample_indices_without_replacement_via_tracking_selection(num_total, num_samples, rng)
    elif ratio > 0.5:
        # For large ratios use reservoir sampling
        return __sample_indices_without_replacement_via_reservoir_sampling(num_total, num_samples, rng)
    else:
        # Otherwise, use random permutation as the default method
        return __sample_indices_without_replacement_via_random_permutation(num_total, num_samples, rng)


cdef inline intp[::1] __sample_indices_without_replacement_via_tracking_selection(intp num_total, intp num_samples,
                                                                                  RNG rng):
    """
    Randomly selects `num_samples` out of `num_total` indices without replacement by using a set to keep track of the
    indices that have already been selected. This method is suitable if `num_samples` is much smaller than `num_total`.

    :param num_total:   The total number of available indices
    :param num_samples: The number of indices to be sampled
    :param rng:         The random number generator to be used
    :return:            An array of dtype int, shape `(num_samples)`, representing the indices contained in the
                        sub-sample
    """
    cdef intp[::1] indices = array_intp(num_samples)
    cdef set[uint32] selected_indices  # Stack-allocated set
    cdef bint should_continue
    cdef uint32 random_index
    cdef intp i

    for i in range(num_samples):
        should_continue = True

        while should_continue:
            random_index = rng.random(0, num_total)
            should_continue = not selected_indices.insert(random_index).second

        indices[i] = random_index

    return indices


cdef inline intp[::1] __sample_indices_without_replacement_via_reservoir_sampling(intp num_total, intp num_samples,
                                                                                  RNG rng):
    """
    Randomly selects `num_samples` out of `num_total` indices without replacement using a reservoir sampling algorithm.
    This method is suitable if `num_samples` is almost as large as `num_total`.

    :param num_total:   The total number of available indices
    :param num_samples: The number of indices to be sampled
    :param rng:         The random number generator to be used
    :return:            An array of dtype int, shape `(num_samples)`, representing the indices contained in the
                        sub-sample
    """
    cdef intp[::1] indices = array_intp(num_samples)
    cdef uint32 random_index
    cdef intp i

    for i in range(num_samples):
        indices[i] = i

    for i from num_samples <= i < num_total:
        random_index = rng.random(0, i + 1)

        if random_index < num_samples:
            indices[random_index] = i

    return indices


cdef inline intp[::1] __sample_indices_without_replacement_via_random_permutation(intp num_total, intp num_samples,
                                                                                  RNG rng):
    """
    Randomly selects `num_samples` out of `num_total` indices without replacement by first generating a random
    permutation of the available indices using the Fisher-Yates shuffle and then returning the first `num_samples`
    indices.

    :param num_total:   The total number of available indices
    :param num_samples: The number of indices to be sampled
    :param rng:         The random number generator to be used
    :return:            An array of dtype int, shape `(num_samples)`, representing the indices contained in the
                        sub-sample
    """
    cdef intp[::1] indices = array_intp(num_total)
    cdef uint32 random_index
    cdef intp i, tmp

    for i in range(num_total):
        indices[i] = i

    for i in range(num_total - 2):
        # Swap elements at i and a randomly selected index...
        random_index = rng.random(i, num_total)
        tmp = indices[i]
        indices[i] = indices[random_index]
        indices[random_index] = tmp

    return indices[:num_samples]
