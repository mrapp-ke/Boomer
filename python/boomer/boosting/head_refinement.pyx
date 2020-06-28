"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes that implement strategies for finding the heads of rules, which are specific to boosting algorithms.
"""
from boomer.common._arrays cimport float64, array_float64


cdef class FullHeadRefinement(HeadRefinement):
    """
    Allows to find the best multi-label head that predicts for all labels.
    """

    cdef HeadCandidate find_head(self, HeadCandidate best_head, intp[::1] label_indices, Loss loss, bint uncovered,
                                 bint accumulated):
        cdef Prediction prediction = loss.evaluate_label_dependent_predictions(uncovered, accumulated)
        cdef float64[::1] predicted_scores = prediction.predicted_scores
        cdef float64 overall_quality_score = prediction.overall_quality_score
        cdef intp num_labels = predicted_scores.shape[0]
        cdef float64[::1] candidate_predicted_scores
        cdef HeadCandidate candidate
        cdef intp c

        if best_head is None:
            # Create a new `HeadCandidate` and return it...
            candidate_predicted_scores = array_float64(num_labels)

            for c in range(num_labels):
                candidate_predicted_scores[c] = predicted_scores[c]

            candidate = HeadCandidate.__new__(HeadCandidate, label_indices, candidate_predicted_scores,
                                              overall_quality_score)
            return candidate
        else:
            # The quality score must be better than that of `best_head`...
            if overall_quality_score < best_head.quality_score:
                # Modify the `best_head` and return it...
                for c in range(num_labels):
                    best_head.predicted_scores[c] = predicted_scores[c]

                best_head.quality_score = overall_quality_score
                return best_head

        # Return None, as the quality score of the found head is worse than that of `best_head`...
        return None

    cdef Prediction evaluate_predictions(self, Loss loss, bint uncovered, bint accumulated):
        cdef Prediction prediction = loss.evaluate_label_dependent_predictions(uncovered, accumulated)
        return prediction
