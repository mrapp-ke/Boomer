from boomer.common._arrays cimport intp
from boomer.common.losses cimport Loss, Prediction
from boomer.common.head_refinement cimport HeadRefinement, HeadCandidate


cdef class FullHeadRefinement(HeadRefinement):

    # Functions:

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered,
                                 bint accumulated)

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered, bint accumulated)