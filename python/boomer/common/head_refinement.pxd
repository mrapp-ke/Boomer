from boomer.common._arrays cimport intp, float64
from boomer.common.losses cimport Loss, Prediction


cdef class HeadCandidate:

    # Attributes:

    cdef readonly intp[::1] label_indices

    cdef readonly float64[::1] predicted_scores

    cdef readonly float64 quality_score


cdef class HeadRefinement:

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered,
                                 bint accumulated)

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered, bint accumulated)


cdef class SingleLabelHeadRefinement(HeadRefinement):

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered,
                                 bint accumulated)

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered, bint accumulated)
