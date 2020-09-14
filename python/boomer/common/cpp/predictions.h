/**
 * Provides classes that store the predictions of rules, as well as corresponding quality scores.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"


/**
 * Stores the predictions of a rule for several labels.
 */
class Prediction {

    public:

        /**
         * @param numPredictions    The number of labels for which the rule predicts
         * @param labelIndices      A pointer to an array of type `uint32`, shape `(numPredictions)`, representing the
         *                          indices of the labels for which the rule predicts of NULL, if the rule predicts for
         *                          all labels
         * @param predictedScores   A pointer to an array of type `float64`, shape `(numPredictions)`, representing the
         *                          predicted scores
         */
        Prediction(uint32 numPredictions, uint32* labelIndices, float64* predictedScores);

        ~Prediction();

        /**
         * The number of labels for which the rule predicts.
         */
        uint32 numPredictions_;

        /**
         * A pointer to an array of type `uint32`, shape `(numPredictions_)`, representing the indices of the labels for
         * which the rule predicts or NULL, if the rule predicts for all labels.
         */
        uint32* labelIndices_;

        /**
         * A pointer to an array of type `float64`, shape `(numPredictions_)`, representing the predicted scores.
         */
        float64* predictedScores_;

};

/**
 * Stores the predictions of a rule for several labels, as well as an overall quality score.
 */
class PredictionCandidate : public Prediction {

    public:

        /**
         * @param numPredictions        The number of labels for which the rule predicts
         * @param labelIndices          A pointer to an array of type `uint32`, shape `(numPredictions)`, representing
         *                              the indices of the labels for which the rule predicts of NULL, if the rule
         *                              predicts for all labels
         * @param predictedScores       A pointer to an array of type `float64`, shape `(numPredictions)`, representing
         *                              the predicted scores
         * @param overallQualityScore   A score that assesses the overall quality of the predictions
         */
        PredictionCandidate(uint32 numPredictions, uint32* labelIndices, float64* predictedScores,
                            float64 overallQualityScore);

        ~PredictionCandidate();

        /**
         * A score that assesses the overall quality of the predictions.
         */
        float64 overallQualityScore_;

};

/**
 * Stores the predictions of a rule for several labels, as well as corresponding quality scores.
 */
class LabelWisePredictionCandidate : public PredictionCandidate {

    public:

        /**
         * @param numPredictions        The number of labels for which the rule predicts
         * @param labelIndices          A pointer to an array of type `uint32`, shape `(numPredictions)`, representing
         *                              the indices of the labels for which the rule predicts of NULL, if the rule
         *                              predicts for all labels
         * @param predictedScores       A pointer to an array of type `float64`, shape `(numPredictions)`, representing
         *                              the predicted scores
         * @param qualityScores         A pointer to an array of type `float64`, shape `(numPredictions)`, representing
         *                              the quality scores for individual labels
         * @param overallQualityScore   A score that assesses the overall quality of the predictions
         */
        LabelWisePredictionCandidate(uint32 numPredictions, uint32* labelIndices, float64* predictedScores,
                                     float64* qualityScores, float64 overallQualityScore);

        ~LabelWisePredictionCandidate();

        /**
         * A pointer to an array of type `float64`, shape `(numPredictions_)`, representing the quality scores for
         * individual labels.
         */
        float64* qualityScores_;

};
