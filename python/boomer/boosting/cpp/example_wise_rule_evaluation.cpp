#include "example_wise_rule_evaluation.h"
#include "linalg.h"
#include <cstddef>
#include <stdlib.h>
#include <math.h>

using namespace boosting;


AbstractExampleWiseRuleEvaluation::~AbstractExampleWiseRuleEvaluation() {

}

void AbstractExampleWiseRuleEvaluation::calculateLabelWisePrediction(const uint32* labelIndices,
                                                                     const float64* totalSumsOfGradients,
                                                                     float64* sumsOfGradients,
                                                                     const float64* totalSumsOfHessians,
                                                                     float64* sumsOfHessians, bool uncovered,
                                                                     LabelWisePredictionCandidate* prediction) {

}

void AbstractExampleWiseRuleEvaluation::calculateExampleWisePrediction(const uint32* labelIndices,
                                                                       const float64* totalSumsOfGradients,
                                                                       float64* sumsOfGradients,
                                                                       const float64* totalSumsOfHessians,
                                                                       float64* sumsOfHessians, float64* tmpGradients,
                                                                       float64* tmpHessians, int dsysvLwork,
                                                                       float64* dsysvTmpArray1, int* dsysvTmpArray2,
                                                                       double* dsysvTmpArray3, float64* dspmvTmpArray,
                                                                       bool uncovered,
                                                                       PredictionCandidate* prediction) {

}

RegularizedExampleWiseRuleEvaluationImpl::RegularizedExampleWiseRuleEvaluationImpl(float64 l2RegularizationWeight,
                                                                                   std::shared_ptr<Blas> blasPtr,
                                                                                   std::shared_ptr<Lapack> lapackPtr) {
    l2RegularizationWeight_ = l2RegularizationWeight;
    blasPtr_ = blasPtr;
    lapackPtr_ = lapackPtr;
}

RegularizedExampleWiseRuleEvaluationImpl::~RegularizedExampleWiseRuleEvaluationImpl() {

}

void RegularizedExampleWiseRuleEvaluationImpl::calculateLabelWisePrediction(const uint32* labelIndices,
                                                                            const float64* totalSumsOfGradients,
                                                                            float64* sumsOfGradients,
                                                                            const float64* totalSumsOfHessians,
                                                                            float64* sumsOfHessians, bool uncovered,
                                                                            LabelWisePredictionCandidate* prediction) {
    // Class members
    float64 l2RegularizationWeight = l2RegularizationWeight_;
    // The number of elements in the arrays `predictedScores` and `qualityScores`
    uint32 numPredictions = prediction->numPredictions_;
    // The array that should be used to store the predicted scores
    float64* predictedScores = prediction->predictedScores_;
    // The array that should be used to store the quality scores
    float64* qualityScores = prediction->qualityScores_;
    // The overall quality score, i.e. the sum of the quality scores for each label plus the L2 regularization term
    float64 overallQualityScore = 0;

    // To avoid array recreation each time this function is called, the array for storing the quality scores is only
    // initialized if it has not been initialized yet
    if (qualityScores == NULL) {
        qualityScores = (float64*) malloc(numPredictions * sizeof(float64));
        prediction->qualityScores_ = qualityScores;
    }

    // For each label, calculate the score to be predicted, as well as a quality score...
    for (uint32 c = 0; c < numPredictions; c++) {
        float64 sumOfGradients = sumsOfGradients[c];
        uint32 c2 = linalg::triangularNumber(c + 1) - 1;
        float64 sumOfHessians = sumsOfHessians[c2];

        if (uncovered) {
            uint32 l = labelIndices != NULL ? labelIndices[c] : c;
            sumOfGradients = totalSumsOfGradients[l] - sumOfGradients;
            uint32 l2 = linalg::triangularNumber(l + 1) - 1;
            sumOfHessians = totalSumsOfHessians[l2] - sumOfHessians;
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

void RegularizedExampleWiseRuleEvaluationImpl::calculateExampleWisePrediction(const uint32* labelIndices,
                                                                              const float64* totalSumsOfGradients,
                                                                              float64* sumsOfGradients,
                                                                              const float64* totalSumsOfHessians,
                                                                              float64* sumsOfHessians,
                                                                              float64* tmpGradients,
                                                                              float64* tmpHessians, int dsysvLwork,
                                                                              float64* dsysvTmpArray1,
                                                                              int* dsysvTmpArray2,
                                                                              double* dsysvTmpArray3,
                                                                              float64* dspmvTmpArray,
                                                                              bool uncovered,
                                                                              PredictionCandidate* prediction) {
    // Class members
    float64 l2RegularizationWeight = l2RegularizationWeight_;
    // The number of elements in the arrays `predictedScores`
    uint32 numPredictions = prediction->numPredictions_;
    // The array that should be used to store the predicted scores
    float64* predictedScores = prediction->predictedScores_;

    float64* gradients;
    float64* hessians;

    if (uncovered) {
        gradients = tmpGradients;
        hessians = tmpHessians;
        uint32 i = 0;

        for (uint32 c = 0; c < numPredictions; c++) {
            uint32 l = labelIndices != NULL ? labelIndices[c] : c;
            gradients[c] = totalSumsOfGradients[l] - sumsOfGradients[c];
            uint32 offset = linalg::triangularNumber(l);

            for (uint32 c2 = 0; c2 < c + 1; c2++) {
                uint32 l2 = offset + (labelIndices != NULL ? labelIndices[c2] : c2);
                hessians[i] = totalSumsOfHessians[l2] - sumsOfHessians[i];
                i++;
            }
        }
    } else {
        gradients = sumsOfGradients;
        hessians = sumsOfHessians;
    }

    // Calculate the scores to be predicted for the individual labels by solving a system of linear equations...
    lapackPtr_.get()->dsysv(hessians, gradients, dsysvTmpArray1, dsysvTmpArray2, dsysvTmpArray3, predictedScores,
                            numPredictions, dsysvLwork, l2RegularizationWeight);

    // Calculate overall quality score as (gradients * scores) + (0.5 * (scores * (hessians * scores)))...
    float64 overallQualityScore = blasPtr_.get()->ddot(predictedScores, gradients, numPredictions);
    blasPtr_.get()->dspmv(hessians, predictedScores, dspmvTmpArray, numPredictions);
    overallQualityScore += 0.5 * blasPtr_.get()->ddot(predictedScores, dspmvTmpArray, numPredictions);

    // Add the L2 regularization term to the overall quality score...
    overallQualityScore += 0.5 * l2RegularizationWeight * linalg::l2NormPow(predictedScores, numPredictions);
    prediction->overallQualityScore_ = overallQualityScore;
}
