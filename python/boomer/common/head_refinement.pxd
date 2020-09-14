from boomer.common._arrays cimport uint32
from boomer.common._predictions cimport PredictionCandidate
from boomer.common.statistics cimport AbstractRefinementSearch


cdef class HeadRefinement:

    # Functions:

    cdef PredictionCandidate* find_head(self, PredictionCandidate* best_head, PredictionCandidate* recyclable_head,
                                        const uint32* label_indices, AbstractRefinementSearch* refinement_search,
                                        bint uncovered, bint accumulated) nogil

    cdef PredictionCandidate* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                                   bint accumulated) nogil


cdef class SingleLabelHeadRefinement(HeadRefinement):

    # Functions:

    cdef PredictionCandidate* find_head(self, PredictionCandidate* best_head, PredictionCandidate* recyclable_head,
                                        const uint32* label_indices, AbstractRefinementSearch* refinement_search,
                                        bint uncovered, bint accumulated) nogil

    cdef PredictionCandidate* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                                   bint accumulated) nogil


cdef class FullHeadRefinement(HeadRefinement):

    # Functions:

    cdef PredictionCandidate* find_head(self, PredictionCandidate* best_head, PredictionCandidate* recyclable_head,
                                        const uint32* label_indices, AbstractRefinementSearch* refinement_search,
                                        bint uncovered, bint accumulated) nogil

    cdef PredictionCandidate* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                                   bint accumulated) nogil
