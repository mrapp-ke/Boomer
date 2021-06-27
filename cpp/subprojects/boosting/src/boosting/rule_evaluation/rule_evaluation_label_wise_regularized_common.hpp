/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/math/math.hpp"


namespace boosting {

    template<class ScoreIterator, class QualityScoreIterator, class GradientIterator, class HessianIterator>
    static inline float64 calculateLabelWisePredictionInternally(uint32 numPredictions, ScoreIterator scoreIterator,
                                                                 QualityScoreIterator qualityScoreIterator,
                                                                 GradientIterator gradientIterator,
                                                                 HessianIterator hessianIterator,
                                                                 float64 l2RegularizationWeight) {
        float64 overallQualityScore = 0;

        // For each label, calculate a score to be predicted, as well as a corresponding quality score...
        for (uint32 c = 0; c < numPredictions; c++) {
            float64 sumOfGradients = gradientIterator[c];
            float64 sumOfHessians = hessianIterator[c];

            // Calculate the score to be predicted for the current label...
            float64 score = sumOfHessians + l2RegularizationWeight;
            score = divideOrZero<float64>(-sumOfGradients, score);
            scoreIterator[c] = score;

            // Calculate the quality score for the current label...
            float64 scorePow = score * score;
            score = (sumOfGradients * score) + (0.5 * scorePow * sumOfHessians);
            qualityScoreIterator[c] = score + (0.5 * l2RegularizationWeight * scorePow);
            overallQualityScore += score;
        }

        // Add the L2 regularization term to the overall quality score...
        overallQualityScore += 0.5 * l2RegularizationWeight * l2NormPow<ScoreIterator>(scoreIterator, numPredictions);
        return overallQualityScore;
    }

}
