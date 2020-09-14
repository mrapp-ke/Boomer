#include "label_wise_rule_evaluation.h"
#include "linalg.h"
#include <cstddef>
#include <math.h>

using namespace boosting;


AbstractLabelWiseRuleEvaluation::~AbstractLabelWiseRuleEvaluation() {

}

void AbstractLabelWiseRuleEvaluation::calculateLabelWisePrediction(const uint32* labelIndices,
                                                                   const float64* totalSumsOfGradients,
                                                                   float64* sumsOfGradients,
                                                                   const float64* totalSumsOfHessians,
                                                                   float64* sumsOfHessians, bool uncovered,
                                                                   LabelWisePredictionCandidate* prediction) {

}

RegularizedLabelWiseRuleEvaluationImpl::RegularizedLabelWiseRuleEvaluationImpl(float64 l2RegularizationWeight) {
    l2RegularizationWeight_ = l2RegularizationWeight;
}

RegularizedLabelWiseRuleEvaluationImpl::~RegularizedLabelWiseRuleEvaluationImpl() {

}

void RegularizedLabelWiseRuleEvaluationImpl::calculateLabelWisePrediction(const uint32* labelIndices,
                                                                          const float64* totalSumsOfGradients,
                                                                          float64* sumsOfGradients,
                                                                          const float64* totalSumsOfHessians,
                                                                          float64* sumsOfHessians, bool uncovered,
                                                                          LabelWisePredictionCandidate* prediction) {
    // Class members
    float64 l2RegularizationWeight = l2RegularizationWeight_;
    // The number of labels to predict for
    uint32 numPredictions = prediction->numPredictions_;
    // The array that should be used to store the predicted scores
    float64* predictedScores = prediction->predictedScores_;
    // The array that should be used to store the quality scores
    float64* qualityScores = prediction->qualityScores_;
    // The overall quality score, i.e., the sum of the quality scores for each label plus the L2 regularization term
    float64 overallQualityScore = 0;

    // For each label, calculate a score to be predicted, as well as a corresponding quality score...
    for (uint32 c = 0; c < numPredictions; c++) {
        float64 sumOfGradients = sumsOfGradients[c];
        float64 sumOfHessians =  sumsOfHessians[c];

        if (uncovered) {
            uint32 l = labelIndices != NULL ? labelIndices[c] : c;
            sumOfGradients = totalSumsOfGradients[l] - sumOfGradients;
            sumOfHessians = totalSumsOfHessians[l] - sumOfHessians;
        }

        // Calculate the score to be predicted for the current label...
        float64 score = sumOfHessians + l2RegularizationWeight;
        score = score != 0 ? -sumOfGradients / score : 0;
        predictedScores[c] = score;

        // Calculate the quality score for the current label...
        float64 scorePow = pow(score, 2);
        score = (sumOfGradients * score) + (0.5 * scorePow * sumOfHessians);
        qualityScores[c] = score + (0.5 * l2RegularizationWeight * scorePow);
        overallQualityScore += score;
    }

    // Add the L2 regularization term to the overall quality score...
    overallQualityScore += 0.5 * l2RegularizationWeight * linalg::l2NormPow(predictedScores, numPredictions);
    prediction->overallQualityScore_ = overallQualityScore;
}
