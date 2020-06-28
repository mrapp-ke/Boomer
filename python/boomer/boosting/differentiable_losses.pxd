from boomer.common._arrays cimport uint8, uint32, intp, float64
from boomer.common.losses cimport Loss, Prediction, LabelIndependentPrediction

from libc.math cimport pow


cdef class DifferentiableLoss(Loss):

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef void begin_instance_sub_sampling(self)

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove)

    cdef void begin_search(self, intp[::1] label_indices)

    cdef void update_search(self, intp example_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered, bint accumulated)

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered, bint accumulated)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores)


cdef class DecomposableDifferentiableLoss(DifferentiableLoss):

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef void begin_instance_sub_sampling(self)

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove)

    cdef void begin_search(self, intp[::1] label_indices)

    cdef void update_search(self, intp example_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered, bint accumulated)

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered, bint accumulated)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores)


cdef class NonDecomposableDifferentiableLoss(DifferentiableLoss):

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef void begin_instance_sub_sampling(self)

    cdef void update_sub_sample(self, intp example_index, uint32 weight, bint remove)

    cdef void begin_search(self, intp[::1] label_indices)

    cdef void update_search(self, intp example_index, uint32 weight)

    cdef void reset_search(self)

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered, bint accumulated)

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered, bint accumulated)

    cdef void apply_prediction(self, intp example_index, intp[::1] label_indices, float64[::1] predicted_scores)


cdef inline float64 _convert_label_into_score(uint8 label):
    """
    Converts a label {0, 1} into an expected score {-1, 1}.

    :param label:   A scalar of dtype `uint8`, representing the label
    :return:        A scalar of dtype `float64`, representing the expected score
    """
    if label > 0:
        return label
    else:
        return -1


cdef inline float64 _l2_norm_pow(float64[::1] a):
    """
    Computes and returns the square of the L2 norm of a specific vector, i.e. the sum of the squares of its elements. To
    obtain the actual L2 norm, the square-root of the result provided by this function must be computed.

    :param a:   An array of dtype `float64`, shape (n), representing a vector
    :return:    A scalar of dtype `float64`, representing the square of the L2 of the given vector
    """
    cdef float64 result = 0
    cdef intp n = a.shape[0]
    cdef float64 tmp
    cdef intp i

    for i in range(n):
        tmp = a[i]
        tmp = pow(tmp, 2)
        result += tmp

    return result
