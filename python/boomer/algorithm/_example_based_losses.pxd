from boomer.algorithm._arrays cimport uint8, uint32, intp, float64
from boomer.algorithm._losses cimport NonDecomposableLoss, Prediction, LabelIndependentPrediction


cdef class ExampleBasedLogisticLoss(NonDecomposableLoss):

    # Attributes:

    cdef float64[::1, :] expected_scores

    cdef float64[::1, :] current_scores

    cdef float64[::1, :] gradients

    cdef float64[::1] sums_of_gradients

    cdef float64[::1] total_sums_of_gradients

    cdef float64[::1, :] hessians

    cdef float64[::1] sums_of_hessians

    cdef float64[::1] total_sums_of_hessians

    cdef intp[::1] label_indices

    cdef LabelIndependentPrediction prediction

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