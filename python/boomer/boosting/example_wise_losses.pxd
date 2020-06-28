from boomer.common._arrays cimport uint8, uint32, intp, float64
from boomer.common.losses cimport Prediction, LabelIndependentPrediction
from boomer.boosting.differentiable_losses cimport NonDecomposableDifferentiableLoss


cdef class ExampleWiseLogisticLoss(NonDecomposableDifferentiableLoss):

    # Attributes:

    cdef float64 l2_regularization_weight

    cdef float64[::1, :] expected_scores

    cdef float64[::1, :] current_scores

    cdef float64[::1, :] gradients

    cdef float64[::1] sums_of_gradients

    cdef float64[::1] accumulated_sums_of_gradients

    cdef float64[::1] total_sums_of_gradients

    cdef float64[::1, :] hessians

    cdef float64[::1] sums_of_hessians

    cdef float64[::1] accumulated_sums_of_hessians

    cdef float64[::1] total_sums_of_hessians

    cdef intp[::1] label_indices

    cdef LabelIndependentPrediction prediction

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