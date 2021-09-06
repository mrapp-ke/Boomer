#include "common/rule_refinement/score_processor.hpp"
#include "common/rule_refinement/prediction_complete.hpp"
#include "common/rule_refinement/prediction_partial.hpp"
#include <algorithm>


template<typename T>
const AbstractEvaluatedPrediction* processCompleteScores(std::unique_ptr<AbstractEvaluatedPrediction>& existingHeadPtr,
                                                         const T& scoreVector) {
    if (existingHeadPtr.get() == nullptr) {
        // Create a new head, if necessary...
        uint32 numElements = scoreVector.getNumElements();
        existingHeadPtr = std::make_unique<CompletePrediction>(numElements);
    }

    std::copy(scoreVector.scores_cbegin(), scoreVector.scores_cend(), existingHeadPtr->scores_begin());
    existingHeadPtr->overallQualityScore = scoreVector.overallQualityScore;
    return existingHeadPtr.get();
}

template<typename T>
const AbstractEvaluatedPrediction* processPartialScores(std::unique_ptr<AbstractEvaluatedPrediction>& existingHeadPtr,
                                                        const T& scoreVector) {
    PartialPrediction* existingHead = (PartialPrediction*) existingHeadPtr.get();

    if (existingHead == nullptr) {
        // Create a new head, if necessary...
        uint32 numElements = scoreVector.getNumElements();
        existingHeadPtr = std::make_unique<PartialPrediction>(numElements);
        existingHead = (PartialPrediction*) existingHeadPtr.get();
    } else {
        // Adjust the size of the existing head, if necessary...
        uint32 numElements = scoreVector.getNumElements();

        if (existingHead->getNumElements() != numElements) {
            existingHead->setNumElements(numElements, false);
        }
    }

    std::copy(scoreVector.scores_cbegin(), scoreVector.scores_cend(), existingHead->scores_begin());
    std::copy(scoreVector.indices_cbegin(), scoreVector.indices_cend(), existingHead->indices_begin());
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
