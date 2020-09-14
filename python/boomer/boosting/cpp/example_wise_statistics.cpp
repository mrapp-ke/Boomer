#include "example_wise_statistics.h"
#include "linalg.h"
#include <stdlib.h>
#include <cstddef>

using namespace boosting;


DenseExampleWiseRefinementSearchImpl::DenseExampleWiseRefinementSearchImpl(
        std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr, std::shared_ptr<Lapack> lapackPtr,
        uint32 numPredictions, const uint32* labelIndices, uint32 numLabels, const float64* gradients,
        const float64* totalSumsOfGradients, const float64* hessians, const float64* totalSumsOfHessians) {
    ruleEvaluationPtr_ = ruleEvaluationPtr;
    lapackPtr_ = lapackPtr;
    numPredictions_ = numPredictions;
    labelIndices_ = labelIndices;
    numLabels_ = numLabels;
    gradients_ = gradients;
    totalSumsOfGradients_ = totalSumsOfGradients;
    float64* sumsOfGradients = (float64*) malloc(numPredictions * sizeof(float64));
    arrays::setToZeros(sumsOfGradients, numPredictions);
    sumsOfGradients_ = sumsOfGradients;
    accumulatedSumsOfGradients_ = NULL;
    hessians_ = hessians;
    totalSumsOfHessians_ = totalSumsOfHessians;
    uint32 numHessians = linalg::triangularNumber(numPredictions);
    float64* sumsOfHessians = (float64*) malloc(numHessians * sizeof(float64));
    arrays::setToZeros(sumsOfHessians, numHessians);
    sumsOfHessians_ = sumsOfHessians;
    accumulatedSumsOfHessians_ = NULL;
    float64* predictedScores = (float64*) malloc(numPredictions * sizeof(float64));
    prediction_ = new LabelWisePredictionCandidate(numPredictions, NULL, predictedScores, NULL, 0);
    tmpGradients_ = NULL;
    tmpHessians_ = NULL;
    dsysvTmpArray1_ = NULL;
    dsysvTmpArray2_ = NULL;
    dsysvTmpArray3_ = NULL;
    dspmvTmpArray_ = NULL;
}

DenseExampleWiseRefinementSearchImpl::~DenseExampleWiseRefinementSearchImpl() {
    free(sumsOfGradients_);
    free(accumulatedSumsOfGradients_);
    free(sumsOfHessians_);
    free(accumulatedSumsOfHessians_);
    free(tmpGradients_);
    free(tmpHessians_);
    free(dsysvTmpArray1_);
    free(dsysvTmpArray2_);
    free(dsysvTmpArray3_);
    free(dspmvTmpArray_);
    delete prediction_;
}

void DenseExampleWiseRefinementSearchImpl::updateSearch(uint32 statisticIndex, uint32 weight) {
    // Add the gradients and Hessians of the example at the given index (weighted by the given weight) to the current
    // sum of gradients and Hessians...
    uint32 offsetGradients = statisticIndex * numLabels_;
    uint32 offsetHessians = statisticIndex * linalg::triangularNumber(numLabels_);
    uint32 i = 0;

    for (uint32 c = 0; c < numPredictions_; c++) {
        uint32 l = labelIndices_ != NULL ? labelIndices_[c] : c;
        sumsOfGradients_[c] += (weight * gradients_[offsetGradients + l]);
        uint32 triangularNumber = linalg::triangularNumber(l);

        for (uint32 c2 = 0; c2 < c + 1; c2++) {
            uint32 l2 = triangularNumber + (labelIndices_ != NULL ? labelIndices_[c2] : c2);
            sumsOfHessians_[i] += (weight * hessians_[offsetHessians + l2]);
            i++;
        }
    }
}

void DenseExampleWiseRefinementSearchImpl::resetSearch() {
    uint32 numHessians = linalg::triangularNumber(numPredictions_);

    // Allocate arrays for storing the accumulated sums of gradients and Hessians, if necessary...
    if (accumulatedSumsOfGradients_ == NULL) {
        accumulatedSumsOfGradients_ = (float64*) malloc(numPredictions_ * sizeof(float64));
        arrays::setToZeros(accumulatedSumsOfGradients_, numPredictions_);
        accumulatedSumsOfHessians_ = (float64*) malloc(numHessians * sizeof(float64));
        arrays::setToZeros(accumulatedSumsOfHessians_, numHessians);
    }

    // Reset the sum of gradients and Hessians for each label to zero and add it to the accumulated sums of gradients
    // and Hessians...
    for (uint32 c = 0; c < numPredictions_; c++) {
        accumulatedSumsOfGradients_[c] += sumsOfGradients_[c];
        sumsOfGradients_[c] = 0;
    }

    for (uint32 c = 0; c < numHessians; c++) {
        accumulatedSumsOfHessians_[c] += sumsOfHessians_[c];
        sumsOfHessians_[c] = 0;
    }
}

LabelWisePredictionCandidate* DenseExampleWiseRefinementSearchImpl::calculateLabelWisePrediction(bool uncovered,
                                                                                                 bool accumulated) {
    float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
    float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;
    ruleEvaluationPtr_.get()->calculateLabelWisePrediction(labelIndices_, totalSumsOfGradients_, sumsOfGradients,
                                                           totalSumsOfHessians_, sumsOfHessians, uncovered,
                                                           prediction_);
    return prediction_;
}

PredictionCandidate* DenseExampleWiseRefinementSearchImpl::calculateExampleWisePrediction(bool uncovered,
                                                                                          bool accumulated) {
    float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
    float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;

    // To avoid array recreation each time this function is called, the temporary arrays are only initialized if they
    // have not been initialized yet
    if (tmpGradients_ == NULL) {
        tmpGradients_ = (float64*) malloc(numPredictions_ * sizeof(float64));
        uint32 numHessians = linalg::triangularNumber(numPredictions_);
        tmpHessians_ = (float64*) malloc(numHessians * sizeof(float64));
        dsysvTmpArray1_ = (float64*) malloc(numPredictions_ * numPredictions_ * sizeof(float64));
        dsysvTmpArray2_ = (int*) malloc(numPredictions_ * sizeof(int));
        dspmvTmpArray_ = (float64*) malloc(numPredictions_ * sizeof(float64));

        // Query the optimal "lwork" parameter to be used by LAPACK'S DSYSV routine...
        dsysvLwork_ = lapackPtr_.get()->queryDsysvLworkParameter(dsysvTmpArray1_, prediction_->predictedScores_,
                                                                 numPredictions_);
        dsysvTmpArray3_ = (double*) malloc(dsysvLwork_ * sizeof(double));
    }

    ruleEvaluationPtr_.get()->calculateExampleWisePrediction(labelIndices_, totalSumsOfGradients_, sumsOfGradients,
                                                             totalSumsOfHessians_, sumsOfHessians, tmpGradients_,
                                                             tmpHessians_, dsysvLwork_, dsysvTmpArray1_,
                                                             dsysvTmpArray2_, dsysvTmpArray3_, dspmvTmpArray_,
                                                             uncovered, prediction_);
    return prediction_;
}

AbstractExampleWiseStatistics::AbstractExampleWiseStatistics(
        uint32 numStatistics, uint32 numLabels, std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr)
    : AbstractGradientStatistics(numStatistics, numLabels) {
    this->setRuleEvaluation(ruleEvaluationPtr);
}

void AbstractExampleWiseStatistics::setRuleEvaluation(
        std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr) {
    ruleEvaluationPtr_ = ruleEvaluationPtr;
}

DenseExampleWiseStatisticsImpl::DenseExampleWiseStatisticsImpl(
        std::shared_ptr<AbstractExampleWiseLoss> lossFunctionPtr,
        std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr, std::shared_ptr<Lapack> lapackPtr,
        std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr, float64* gradients, float64* hessians,
        float64* currentScores)
    : AbstractExampleWiseStatistics(labelMatrixPtr.get()->numExamples_, labelMatrixPtr.get()->numLabels_,
                                    ruleEvaluationPtr) {
    lossFunctionPtr_ = lossFunctionPtr;
    lapackPtr_ = lapackPtr;
    labelMatrixPtr_ = labelMatrixPtr;
    gradients_ = gradients;
    hessians_ = hessians;
    currentScores_ = currentScores;
    // The number of hessians
    uint32 numHessians = linalg::triangularNumber(numLabels_);
    // An array that stores the column-wise sums of the matrix of gradients
    totalSumsOfGradients_ = (float64*) malloc(numLabels_ * sizeof(float64));
    // An array that stores the column-wise sums of the matrix of Hessians
    totalSumsOfHessians_ = (float64*) malloc(numHessians * sizeof(float64));
}

DenseExampleWiseStatisticsImpl::~DenseExampleWiseStatisticsImpl() {
    free(currentScores_);
    free(gradients_);
    free(totalSumsOfGradients_);
    free(hessians_);
    free(totalSumsOfHessians_);
}

void DenseExampleWiseStatisticsImpl::resetCoveredStatistics() {
    arrays::setToZeros(totalSumsOfGradients_, numLabels_);
    uint32 numHessians = linalg::triangularNumber(numLabels_);
    arrays::setToZeros(totalSumsOfHessians_, numHessians);
}

void DenseExampleWiseStatisticsImpl::updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) {
    float64 signedWeight = remove ? -((float64) weight) : weight;
    uint32 offset = statisticIndex * numLabels_;

    // Add the gradients of the example at the given index (weighted by the given weight) to the total sums of
    // gradients...
    for (uint32 c = 0; c < numLabels_; c++) {
        totalSumsOfGradients_[c] += (signedWeight * gradients_[offset + c]);
    }

    uint32 numHessians = linalg::triangularNumber(numLabels_);
    offset = statisticIndex * numHessians;

    // Add the Hessians of the example at the given index (weighted by the given weight) to the total sums of
    // Hessians...
    for (uint32 c = 0; c < numHessians; c++) {
        totalSumsOfHessians_[c] += (signedWeight * hessians_[offset + c]);
    }
}

AbstractRefinementSearch* DenseExampleWiseStatisticsImpl::beginSearch(uint32 numLabelIndices,
                                                                      const uint32* labelIndices) {
    uint32 numPredictions = labelIndices == NULL ? numLabels_ : numLabelIndices;
    return new DenseExampleWiseRefinementSearchImpl(ruleEvaluationPtr_, lapackPtr_, numPredictions, labelIndices,
                                                    numLabels_, gradients_, totalSumsOfGradients_, hessians_,
                                                    totalSumsOfHessians_);
}

void DenseExampleWiseStatisticsImpl::applyPrediction(uint32 statisticIndex, Prediction* prediction) {
    AbstractExampleWiseLoss* lossFunction = lossFunctionPtr_.get();
    uint32 numPredictions = prediction->numPredictions_;
    const uint32* labelIndices = prediction->labelIndices_;
    const float64* predictedScores = prediction->predictedScores_;
    uint32 offset = statisticIndex * numLabels_;
    uint32 numHessians = linalg::triangularNumber(numLabels_);

    // Traverse the labels for which the new rule predicts to update the scores that are currently predicted for the
    // example at the given index...
    for (uint32 c = 0; c < numPredictions; c++) {
        uint32 l = labelIndices != NULL ? labelIndices[c] : c;
        currentScores_[offset + l] += predictedScores[c];
    }

    // Update the gradients and Hessians for the example at the given index...
    lossFunction->calculateGradientsAndHessians(labelMatrixPtr_.get(), statisticIndex, &currentScores_[offset],
                                                &gradients_[offset], &hessians_[statisticIndex * numHessians]);
}

AbstractExampleWiseStatisticsFactory::~AbstractExampleWiseStatisticsFactory() {

}

AbstractExampleWiseStatistics* AbstractExampleWiseStatisticsFactory::create() {
    return NULL;
}

DenseExampleWiseStatisticsFactoryImpl::DenseExampleWiseStatisticsFactoryImpl(
        std::shared_ptr<AbstractExampleWiseLoss> lossFunctionPtr,
        std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr, std::shared_ptr<Lapack> lapackPtr,
        std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr) {
    lossFunctionPtr_ = lossFunctionPtr;
    ruleEvaluationPtr_ = ruleEvaluationPtr;
    lapackPtr_ = lapackPtr;
    labelMatrixPtr_ = labelMatrixPtr;
}

DenseExampleWiseStatisticsFactoryImpl::~DenseExampleWiseStatisticsFactoryImpl() {

}

AbstractExampleWiseStatistics* DenseExampleWiseStatisticsFactoryImpl::create() {
    // Class members
    AbstractExampleWiseLoss* lossFunction = lossFunctionPtr_.get();
    AbstractRandomAccessLabelMatrix* labelMatrix = labelMatrixPtr_.get();
    // The number of examples
    uint32 numExamples = labelMatrix->numExamples_;
    // The number of labels
    uint32 numLabels = labelMatrix->numLabels_;
    // The number of hessians
    uint32 numHessians = linalg::triangularNumber(numLabels);
    // A matrix that stores the gradients for each example
    float64* gradients = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // A matrix that stores the Hessians for each example
    float64* hessians = (float64*) malloc(numExamples * numHessians * sizeof(float64));
    // A matrix that stores the currently predicted scores for each example and label
    float64* currentScores = (float64*) malloc(numExamples * numLabels * sizeof(float64));

    for (uint32 r = 0; r < numExamples; r++) {
        uint32 offset = r * numLabels;

        for (uint32 c = 0; c < numLabels; c++) {
            // Store the score that is initially predicted for the current example and label...
            currentScores[offset + c] = 0;
        }

        // Calculate the initial gradients and Hessians for the current example...
        lossFunction->calculateGradientsAndHessians(labelMatrix, r, &currentScores[offset], &gradients[offset],
                                                    &hessians[r * numHessians]);
    }

    return new DenseExampleWiseStatisticsImpl(lossFunctionPtr_, ruleEvaluationPtr_, lapackPtr_, labelMatrixPtr_,
                                              gradients, hessians, currentScores);
}
