# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides commonly used utility functions and structs.
"""
from boomer.algorithm._arrays cimport intp, uint8, uint32, float32, float64


"""
A struct that represents a condition of a rule. It consists of the index of the feature that is used by the condition,
whether it uses the <= (leq=1) or > (leq=0) operator, as well as a threshold.
"""
cdef struct s_condition:
    intp feature_index
    bint leq
    float32 threshold


cdef inline s_condition make_condition(intp feature_index, bint leq, float32 threshold):
    """
    Creates and returns a new condition.

    :param feature_index:   The index of the feature that is used by the condition
    :param leq:             Whether the <= operator, or the > operator is used by the condition
    :param threshold:       The threshold that is used by the condition
    """
    cdef s_condition condition
    condition.feature_index = feature_index
    condition.leq = leq
    condition.threshold = threshold
    return condition


cdef inline bint test_condition(float32 threshold, bint leq, float32 feature_value):
    """
    Returns whether a given feature value satisfies a certain condition.

    :param threshold:       The threshold of the condition
    :param leq:             1, if the condition uses the <= operator, 0, if it uses the > operator
    :param feature_value:   The feature value
    :return:                1, if the feature value satisfies the condition, 0 otherwise
    """
    if leq:
        return feature_value <= threshold
    else:
        return feature_value > threshold


cdef inline intp get_index(intp i, intp[::1] indices):
    """
    Retrieves and returns the i-th index from an array of indices, if such an array is available. Otherwise i is
    returned.

    :param i:       The position of the index that should be retrieved
    :param indices: An array of the dtype int, shape `(num_indices)`, representing the indices, or None
    :return:        A scalar of dtype int, representing the i-th index in the given array or i, if the array is None
    """
    if indices is None:
        return i
    else:
        return indices[i]


cdef inline uint32 get_weight(intp i, uint32[::1] weights):
    """
    Retrieves and returns the i-th weight from an array of weights, if such an array is available. Otherwise 1 is
    returned.

    :param i:       The position of the weight that should be retrieved
    :param weights: An array of dtype int, shape `(num_weights)`, representing the weights, or None
    :return:        A scalar of dtype int, representing the i-th weight in the given array or 1, if the array is None
    """
    if weights is None:
        return 1
    else:
        return weights[i]


cdef inline float64 convert_label_into_score(uint8 label):
    """
    Converts a label {0, 1} into an expected score {-1, 1}.

    :param label:   A scalar of dtype `uint8`, representing the label
    :return:        A scalar of dtype `float64`, representing the expected score
    """
    if label > 0:
        return label
    else:
        return -1
