#include "common/rule_refinement/score_processor.hpp"
#include "common/rule_refinement/prediction_complete.hpp"
#include "common/rule_refinement/prediction_partial.hpp"
#include "common/data/arrays.hpp"


template<typename T>
const AbstractEvaluatedPrediction* processCompleteScores(std::unique_ptr<AbstractEvaluatedPrediction>& existingHeadPtr,
                                                         const T& scoreVector) {
    uint32 numElements = scoreVector.getNumElements();

    if (existingHeadPtr.get() == nullptr) {
        // Create a new head, if necessary...
        existingHeadPtr = std::make_unique<CompletePrediction>(numElements);
    }

    copyArray(scoreVector.scores_cbegin(), existingHeadPtr->scores_begin(), numElements);
    existingHeadPtr->overallQualityScore = scoreVector.overallQualityScore;
    return existingHeadPtr.get();
}

template<typename T>
const AbstractEvaluatedPrediction* processPartialScores(std::unique_ptr<AbstractEvaluatedPrediction>& existingHeadPtr,
                                                        const T& scoreVector) {
    PartialPrediction* existingHead = (PartialPrediction*) existingHeadPtr.get();
    uint32 numElements = scoreVector.getNumElements();

    if (existingHead == nullptr) {
        // Create a new head, if necessary...
        existingHeadPtr = std::make_unique<PartialPrediction>(numElements);
        existingHead = (PartialPrediction*) existingHeadPtr.get();
    } else if (existingHead->getNumElements() != numElements) {
        // Adjust the size of the existing head, if necessary...
        existingHead->setNumElements(numElements, false);
    }

    copyArray(scoreVector.scores_cbegin(), existingHead->scores_begin(), numElements);
    copyArray(scoreVector.indices_cbegin(), existingHead->indices_begin(), numElements);
    existingHead->overallQualityScore = scoreVector.overallQualityScore;
    return existingHead;
}

const AbstractEvaluatedPrediction* ScoreProcessor::processScores(
        const DenseScoreVector<CompleteIndexVector>& scoreVector) {
    return processCompleteScores(headPtr_, scoreVector);
}

const AbstractEvaluatedPrediction* ScoreProcessor::processScores(
        const DenseScoreVector<PartialIndexVector>& scoreVector) {
    return processPartialScores(headPtr_, scoreVector);
}

const AbstractEvaluatedPrediction* ScoreProcessor::processScores(
        const DenseBinnedScoreVector<CompleteIndexVector>& scoreVector) {
    return processCompleteScores(headPtr_, scoreVector);
}

const AbstractEvaluatedPrediction* ScoreProcessor::processScores(
        const DenseBinnedScoreVector<PartialIndexVector>& scoreVector) {
    return processPartialScores(headPtr_, scoreVector);
}

const AbstractEvaluatedPrediction* ScoreProcessor::processScores(const IScoreVector& scoreVector) {
    return scoreVector.processScores(*this);
}

std::unique_ptr<AbstractEvaluatedPrediction> ScoreProcessor::pollHead() {
    return std::move(headPtr_);
}
