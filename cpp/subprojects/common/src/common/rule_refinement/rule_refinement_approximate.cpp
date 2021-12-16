#include "common/rule_refinement/rule_refinement_approximate.hpp"
#include "common/rule_refinement/score_processor.hpp"
#include "rule_refinement_common.hpp"


template<typename T>
ApproximateRuleRefinement<T>::ApproximateRuleRefinement(
        const T& labelIndices, uint32 featureIndex, bool nominal, const IWeightVector& weights,
        std::unique_ptr<IRuleRefinementCallback<ThresholdVector, BinWeightVector>> callbackPtr)
    : labelIndices_(labelIndices), featureIndex_(featureIndex), nominal_(nominal), weights_(weights),
      callbackPtr_(std::move(callbackPtr)) {

}

template<typename T>
void ApproximateRuleRefinement<T>::findRefinement(const AbstractEvaluatedPrediction* currentHead) {
    std::unique_ptr<Refinement> refinementPtr = std::make_unique<Refinement>();
    refinementPtr->featureIndex = featureIndex_;
    const AbstractEvaluatedPrediction* bestHead = currentHead;
    ScoreProcessor scoreProcessor;

    // Invoke the callback...
    std::unique_ptr<IRuleRefinementCallback<ThresholdVector, BinWeightVector>::Result> callbackResultPtr =
        callbackPtr_->get();
    const IImmutableStatistics& statistics = callbackResultPtr->statistics_;
    const BinWeightVector& weights = callbackResultPtr->weights_;
    const ThresholdVector& thresholdVector = callbackResultPtr->vector_;
    ThresholdVector::const_iterator thresholdIterator = thresholdVector.cbegin();
    uint32 numBins = thresholdVector.getNumElements();
    uint32 sparseBinIndex = thresholdVector.getSparseBinIndex();
    bool sparse = sparseBinIndex < numBins && weights[sparseBinIndex];

    // Create a new, empty subset of the statistics...
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = labelIndices_.createSubset(statistics);

    for (auto it = thresholdVector.missing_indices_cbegin(); it != thresholdVector.missing_indices_cend(); it++) {
        uint32 i = *it;
        float64 weight = weights_.getWeight(i);
        statisticsSubsetPtr->addToMissing(i, weight);
    }

    // In the following, we start by processing the bins in range [0, sparseBinIndex)...
    bool subsetModified = false;
    int64 firstR = 0;
    int64 r;

    // Traverse bins in ascending order until the first bin with weight > 0 is encountered...
    for (r = 0; r < sparseBinIndex; r++) {
        if (weights[r]) {
            // Add the bin to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr->addToSubset(r, 1);
            subsetModified = true;
            break;
        }
    }

    // Traverse the remaining bins in ascending order...
    if (subsetModified) {
        for (r = r + 1; r < sparseBinIndex; r++) {
            // Do only consider bins that are not empty...
            if (weights[r]) {
                // Find and evaluate the best head for the current refinement, if a condition that uses the <= operator
                // (or the == operator in case of a nominal feature) is used...
                const IScoreVector& scoreVector = statisticsSubsetPtr->calculatePrediction(false, false);

                // If the refinement is better than the current rule...
                if (isBetterThanBestHead(scoreVector, bestHead)) {
                    bestHead = scoreProcessor.processScores(scoreVector);
                    refinementPtr->start = firstR;
                    refinementPtr->end = r;
                    refinementPtr->covered = true;
                    refinementPtr->threshold = thresholdIterator[r - 1];
                    refinementPtr->comparator = nominal_ ? EQ : LEQ;
                }

                // Find and evaluate the best head for the current refinement, if a condition that uses the > operator
                // (or the != operator in case of a nominal feature) is used...
                const IScoreVector& scoreVector2 = statisticsSubsetPtr->calculatePrediction(true, false);

                // If the refinement is better than the current rule...
                if (isBetterThanBestHead(scoreVector2, bestHead)) {
                    bestHead = scoreProcessor.processScores(scoreVector2);
                    refinementPtr->start = firstR;
                    refinementPtr->end = r;
                    refinementPtr->covered = false;
                    refinementPtr->threshold = thresholdIterator[r - 1];
                    refinementPtr->comparator = nominal_ ? NEQ : GR;
                }

                // Reset the subset in case of a nominal feature, as the previous bins will not be covered by the next
                // condition...
                if (nominal_) {
                    statisticsSubsetPtr->resetSubset();
                    firstR = r;
                }

                // Add the bin to the subset to mark it as covered by upcoming refinements...
                statisticsSubsetPtr->addToSubset(r, 1);
            }
        }

        // If any bins have been processed so far and if there is a sparse bin, we must evaluate additional conditions
        // that separate the bins that have been iterated from the remaining ones (including the sparse bin)...
        if (subsetModified && sparse) {
            // Find and evaluate the best head for the current refinement, if a condition that uses the <= operator (or
            // the == operator in case of nominal feature) is used...
            const IScoreVector& scoreVector = statisticsSubsetPtr->calculatePrediction(false, false);

            // If the refinement is better than the current rule...
            if (isBetterThanBestHead(scoreVector, bestHead)) {
                bestHead = scoreProcessor.processScores(scoreVector);
                refinementPtr->start = firstR;
                refinementPtr->end = sparseBinIndex;
                refinementPtr->covered = true;
                refinementPtr->threshold = thresholdIterator[sparseBinIndex - 1];
                refinementPtr->comparator = nominal_ ? EQ : LEQ;
            }

            // Find and evaluate the best head for the current refinement, if a condition that uses the > operator (or
            // the != operator in case of a nominal feature) is used...
            const IScoreVector& scoreVector2 = statisticsSubsetPtr->calculatePrediction(true, false);

            // If the refinement is better than the current rule...
            if (isBetterThanBestHead(scoreVector2, bestHead)) {
                bestHead = scoreProcessor.processScores(scoreVector2);
                refinementPtr->start = firstR;
                refinementPtr->end = sparseBinIndex;
                refinementPtr->covered = false;
                refinementPtr->threshold = thresholdIterator[sparseBinIndex - 1];
                refinementPtr->comparator = nominal_ ? NEQ : GR;
            }
        }

        // Reset the subset, if any bins have been processed...
        statisticsSubsetPtr->resetSubset();
    }

    bool subsetModifiedPrevious = subsetModified;

    // We continue by processing the bins in range (sparseBinIndex, numBins)...
    subsetModified = false;
    firstR = ((int64) numBins) - 1;

    // Traverse bins in descending order until the first bin with weight > 0 is encountered...
    for (r = firstR; r > sparseBinIndex; r--) {
        if (weights[r]) {
            // Add the bin to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr->addToSubset(r, 1);
            subsetModified = true;
            break;
        }
    }

    // Traverse the remaining bins in descending order...
    if (subsetModified) {
        for (r = r - 1; r > sparseBinIndex; r--) {
            // Do only consider bins that are not empty...
            if (weights[r]) {
                // Find and evaluate the best head for the current refinement, if a condition that uses the > operator
                // (or the == operator in case of a nominal feature) is used..
                const IScoreVector& scoreVector = statisticsSubsetPtr->calculatePrediction(false, false);

                // If the refinement is better than the current rule...
                if (isBetterThanBestHead(scoreVector, bestHead)) {
                    bestHead = scoreProcessor.processScores(scoreVector);
                    refinementPtr->start = firstR;
                    refinementPtr->end = r;
                    refinementPtr->covered = true;

                    if (nominal_) {
                        refinementPtr->threshold = thresholdIterator[firstR];
                        refinementPtr->comparator = EQ;
                    } else {
                        refinementPtr->threshold = thresholdIterator[r];
                        refinementPtr->comparator = GR;
                    }
                }

                // Find and evaluate the best head for the current refinement, if a condition that uses the <= operator
                // (or the != operator in case of a nominal feature) is used...
                const IScoreVector& scoreVector2 = statisticsSubsetPtr->calculatePrediction(true, false);

                // If the refinement is better than the current rule...
                if (isBetterThanBestHead(scoreVector2, bestHead)) {
                    bestHead = scoreProcessor.processScores(scoreVector2);
                    refinementPtr->start = firstR;
                    refinementPtr->end = r;
                    refinementPtr->covered = false;

                    if (nominal_) {
                        refinementPtr->threshold = thresholdIterator[firstR];
                        refinementPtr->comparator = NEQ;
                    } else {
                        refinementPtr->threshold = thresholdIterator[r];
                        refinementPtr->comparator = LEQ;
                    }

                }

                // Reset the subset in case of a nominal feature, as the previous bins will not be covered by the next
                // condition...
                if (nominal_) {
                    statisticsSubsetPtr->resetSubset();
                    firstR = r;
                }

                // Add the bin to the subset to mark it as covered by upcoming refinements...
                statisticsSubsetPtr->addToSubset(r, 1);
            }
        }

        // If there is a sparse bin, we must evaluate additional conditions that separate the bins in range
        // (sparseBinIndex, numBins) from the remaining ones...
        if (sparse) {
            // Find and evaluate the best head for the current refinement, if
            const IScoreVector& scoreVector = statisticsSubsetPtr->calculatePrediction(false, false);

            // If the refinement is better than the current rule...
            if (isBetterThanBestHead(scoreVector, bestHead)) {
                bestHead = scoreProcessor.processScores(scoreVector);
                refinementPtr->start = firstR;
                refinementPtr->end = sparseBinIndex;
                refinementPtr->covered = true;

                if (nominal_) {
                    refinementPtr->threshold = thresholdIterator[firstR];
                    refinementPtr->comparator = EQ;
                } else {
                    refinementPtr->threshold = thresholdIterator[sparseBinIndex];
                    refinementPtr->comparator = GR;
                }
            }

            // Find and evaluate the best head for the current refinement, if
            const IScoreVector& scoreVector2 = statisticsSubsetPtr->calculatePrediction(true, false);

            // If the refinement is better than the current rule...
            if (isBetterThanBestHead(scoreVector2, bestHead)) {
                bestHead = scoreProcessor.processScores(scoreVector2);
                refinementPtr->start = firstR;
                refinementPtr->end = sparseBinIndex;
                refinementPtr->covered = false;

                if (nominal_) {
                    refinementPtr->threshold = thresholdIterator[firstR];
                    refinementPtr->comparator = NEQ;
                } else {
                    refinementPtr->threshold = thresholdIterator[sparseBinIndex];
                    refinementPtr->comparator = LEQ;
                }
            }

            // If the feature is nominal and if any bins in the range [0, sparseBinIndex) have been processed earlier,
            // we must test additional conditions that separate the sparse bin from the remaining bins...
            if (nominal_ && subsetModifiedPrevious) {
                // Reset the subset once again to ensure that the accumulated state includes all bins that have been
                // processed so far...
                statisticsSubsetPtr->resetSubset();

                // Find and evaluate the best head for the current refinement, if the condition
                // `f != thresholdIterator[sparseBinIndex]` is used...
                const IScoreVector& scoreVector = statisticsSubsetPtr->calculatePrediction(false, true);

                // If the refinement is better than the current rule...
                if (isBetterThanBestHead(scoreVector, bestHead)) {
                    bestHead = scoreProcessor.processScores(scoreVector);
                    refinementPtr->start = sparseBinIndex;
                    refinementPtr->end = sparseBinIndex + 1;
                    refinementPtr->covered = false;
                    refinementPtr->threshold = thresholdIterator[sparseBinIndex];
                    refinementPtr->comparator = NEQ;
                }

                // Find and evaluate the best head for the current refinement, if the condition
                // `f == thresholdIterator[sparseBinIndex]` is used...
                const IScoreVector& scoreVector2 = statisticsSubsetPtr->calculatePrediction(true, true);

                // If the refinement is better than the current rule...
                if (isBetterThanBestHead(scoreVector2, bestHead)) {
                    bestHead = scoreProcessor.processScores(scoreVector2);
                    refinementPtr->start = sparseBinIndex;
                    refinementPtr->end = sparseBinIndex + 1;
                    refinementPtr->covered = true;
                    refinementPtr->threshold = thresholdIterator[sparseBinIndex];
                    refinementPtr->comparator = EQ;
                }
            }
        }
    }

    refinementPtr->headPtr = scoreProcessor.pollHead();
    refinementPtr_ = std::move(refinementPtr);
}

template<typename T>
std::unique_ptr<Refinement> ApproximateRuleRefinement<T>::pollRefinement() {
    return std::move(refinementPtr_);
}

template class ApproximateRuleRefinement<CompleteIndexVector>;
template class ApproximateRuleRefinement<PartialIndexVector>;
