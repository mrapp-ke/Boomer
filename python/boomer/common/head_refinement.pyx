"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for finding the heads of rules.
"""
from boomer.common._arrays cimport float64
from boomer.common._predictions cimport LabelWisePredictionCandidate

from libc.stdlib cimport malloc


cdef class HeadRefinement:
    """
    A base class for all classes that allow to find the best single- or multi-label head for a rule.
    """

    cdef PredictionCandidate* find_head(self, PredictionCandidate* best_head, PredictionCandidate* recyclable_head,
                                        const uint32* label_indices, AbstractRefinementSearch* refinement_search,
                                        bint uncovered, bint accumulated) nogil:
        """
        Finds and returns the best head for a rule given the predictions that are provided by a
        `AbstractRefinementSearch`.

        The `AbstractRefinementSearch` must have been prepared properly via calls to the function
        `AbstractRefinementSearch#updateSearch`.

        :param best_head:           A pointer to an instance of the C++ class `PredictionCandidate` that corresponds to
                                    the best rule known so far (as found in the previous or current refinement
                                    iteration) or NULL, if no such rule is available yet. The new head must be better
                                    than this one, otherwise it is discarded
        :param recyclable_head:     A pointer to an instance of the C++ class `PredictionCandidate` that may be modified
                                    instead of creating a new instance to avoid unnecessary memory allocations or NULL,
                                    if no such instance is available
        :param label_indices:       A pointer to an array of type `uint32`, shape `(num_predictions)`, representing the
                                    indices of the labels for which the head may predict or NULL, if the head may
                                    predict for all labels
        :param refinement_search:   A pointer to an object of type `AbstractRefinementSearch` to be used for calculating
                                    predictions and corresponding quality scores
        :param uncovered:           0, if the rule for which the head should be found covers all examples that have been
                                    provided to the `AbstractRefinementSearch` so far, 1, if the rule covers all
                                    examples that have not been provided yet
        :param accumulated:         0, if the rule covers all examples that have been provided since the
                                    `AbstractRefinementSearch` has been reset for the last time, 1, if the rule covers
                                    all examples that have been provided so far
        :return:                    A pointer to an instance of the C++ class 'PredictionCandidate' that stores
                                    information about the head that has been found, if the head is better than
                                    `best_head`, NULL otherwise
        """
        pass

    cdef PredictionCandidate* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                                   bint accumulated) nogil:
        """
        Calculates the optimal scores to be predicted by a rule, as well as the rule's overall quality score, using a
        `AbstractRefinementSearch`.

        The `AbstractRefinementSearch` must have been prepared properly via calls to the function
        `AbstractRefinementSearch#updateSearch`.

        :param refinement_search:   A pointer to an object of type `AbstractRefinementSearch` to be used
        :param uncovered:           0, if the rule for which the optimal scores should be calculated covers all examples
                                    that have been provided to the `RefinementSearch` so far, 1, if the rule covers all
                                    examples that have not been provided yet
        :param accumulated          0, if the rule covers all examples that have been provided since the
                                    `RefinementSearch` has been reset for the last time, 1, if the rule covers all
                                    examples that have been  provided so far
        :return:                    A pointer to an object of type `PredictionCandidate` that stores the optimal scores
                                    to be predicted by the rule, as well as its overall quality score
        """
        pass


cdef class SingleLabelHeadRefinement(HeadRefinement):
    """
    Allows to find the best single-label head that predicts for a single label.
    """

    cdef PredictionCandidate* find_head(self, PredictionCandidate* best_head, PredictionCandidate* recyclable_head,
                                        const uint32* label_indices, AbstractRefinementSearch* refinement_search,
                                        bint uncovered, bint accumulated) nogil:
        cdef LabelWisePredictionCandidate* prediction = refinement_search.calculateLabelWisePrediction(uncovered,
                                                                                                       accumulated)
        cdef uint32 num_predictions = prediction.numPredictions_
        cdef float64* predicted_scores = prediction.predictedScores_
        cdef float64* quality_scores = prediction.qualityScores_
        cdef uint32 best_c = 0
        cdef float64 best_quality_score = quality_scores[best_c]
        cdef uint32* candidate_label_indices
        cdef float64* candidate_predicted_scores
        cdef float64 quality_score
        cdef uint32 c

        # Find the best single-label head...
        for c in range(1, num_predictions):
            quality_score = quality_scores[c]

            if quality_score < best_quality_score:
                best_quality_score = quality_score
                best_c = c

        # The quality score must be better than that of `best_head`...
        if best_head == NULL or best_quality_score < best_head.overallQualityScore_:
            if recyclable_head == NULL:
                # Create a new `PredictionCandidate` and return it...
                candidate_label_indices = <uint32*>malloc(sizeof(uint32))
                candidate_label_indices[0] = best_c if label_indices == NULL else label_indices[best_c]
                candidate_predicted_scores = <float64*>malloc(sizeof(float64))
                candidate_predicted_scores[0] = predicted_scores[best_c]
                return new PredictionCandidate(1, candidate_label_indices, candidate_predicted_scores,
                                               best_quality_score)
            else:
                # Modify the `recyclable_head` and return it...
                recyclable_head.labelIndices_[0] = best_c if label_indices == NULL else label_indices[best_c]
                recyclable_head.predictedScores_[0] = predicted_scores[best_c]
                recyclable_head.overallQualityScore_ = best_quality_score
                return recyclable_head

        # Return NULL, as the quality_score of the found head is worse than that of `best_head`...
        return NULL

    cdef PredictionCandidate* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                                   bint accumulated) nogil:
        return refinement_search.calculateLabelWisePrediction(uncovered, accumulated)


cdef class FullHeadRefinement(HeadRefinement):
    """
    Allows to find the best multi-label head that predicts for all labels.
    """

    cdef PredictionCandidate* find_head(self, PredictionCandidate* best_head, PredictionCandidate* recyclable_head,
                                        const uint32* label_indices, AbstractRefinementSearch* refinement_search,
                                        bint uncovered, bint accumulated) nogil:
        cdef PredictionCandidate* prediction = refinement_search.calculateExampleWisePrediction(uncovered, accumulated)
        cdef uint32 num_predictions = prediction.numPredictions_
        cdef float64* predicted_scores = prediction.predictedScores_
        cdef float64 overall_quality_score = prediction.overallQualityScore_
        cdef uint32* candidate_label_indices = NULL
        cdef float64* candidate_predicted_scores
        cdef uint32 c

        # The quality score must be better than that of `best_head`...
        if best_head == NULL or overall_quality_score < best_head.overallQualityScore_:
            if recyclable_head == NULL:
                # Create a new `PredictionCandidate` and return it...
                candidate_predicted_scores = <float64*>malloc(num_predictions * sizeof(float64))

                for c in range(num_predictions):
                    candidate_predicted_scores[c] = predicted_scores[c]

                if label_indices != NULL:
                    candidate_label_indices = <uint32*>malloc(num_predictions * sizeof(uint32))

                    for c in range(num_predictions):
                        candidate_label_indices[c] = label_indices[c]

                return new PredictionCandidate(num_predictions, candidate_label_indices, candidate_predicted_scores,
                                               overall_quality_score)
            else:
                # Modify the `recyclable_head` and return it...
                for c in range(num_predictions):
                    recyclable_head.predictedScores_[c] = predicted_scores[c]

                recyclable_head.overallQualityScore_ = overall_quality_score
                return recyclable_head

        # Return NULL, as the quality score of the found head is worse than that of `best_head`...
        return NULL

    cdef PredictionCandidate* calculate_prediction(self, AbstractRefinementSearch* refinement_search, bint uncovered,
                                                   bint accumulated) nogil:
        return refinement_search.calculateExampleWisePrediction(uncovered, accumulated)
