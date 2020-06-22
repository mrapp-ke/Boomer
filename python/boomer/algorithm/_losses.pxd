from boomer.algorithm._arrays cimport uint8, uint32, intp, float64


cdef class Prediction:

    # Attributes:

    cdef float64[::1] predicted_scores

    cdef float64 overall_quality_score


cdef class LabelIndependentPrediction(Prediction):

    # Attributes:

    cdef float64[::1] quality_scores


cdef class Loss:

    # Attributes:

    cdef readonly float64 l2_regularization_weight

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef begin_instance_sub_sampling(self)

    cdef update_sub_sample(self, intp example_index)

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp example_index, uint32 weight)

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered)

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered)

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores)


cdef class DecomposableLoss(Loss):

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef begin_instance_sub_sampling(self)

    cdef update_sub_sample(self, intp example_index)

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp example_index, uint32 weight)

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered)

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered)

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores)


cdef class NonDecomposableLoss(Loss):

    # Functions:

    cdef float64[::1] calculate_default_scores(self, uint8[::1, :] y)

    cdef begin_instance_sub_sampling(self)

    cdef update_sub_sample(self, intp example_index)

    cdef begin_search(self, intp[::1] label_indices)

    cdef update_search(self, intp example_index, uint32 weight)

    cdef LabelIndependentPrediction evaluate_label_independent_predictions(self, bint uncovered)

    cdef Prediction evaluate_label_dependent_predictions(self, bint uncovered)

    cdef apply_predictions(self, intp[::1] covered_example_indices, intp[::1] label_indices,
                           float64[::1] predicted_scores)
