#include "common/rule_refinement/rule_refinement_approximate.hpp"

#include <iostream>

template<typename IndexVector, typename RefinementComparator>
static inline void findRefinementInternally(const IndexVector& labelIndices, uint32 numExamples, uint32 featureIndex,
                                            bool nominal, uint32 minCoverage,
                                            IRuleRefinementCallback<IHistogram, ThresholdVector>& callback,
                                            RefinementComparator& comparator) {
    Refinement refinement;
    refinement.featureIndex = featureIndex;

    // Invoke the callback...
    IRuleRefinementCallback<IHistogram, ThresholdVector>::Result callbackResult = callback.get();
    const IHistogram& statistics = callbackResult.statistics;
    const ThresholdVector& thresholdVector = callbackResult.vector;
    ThresholdVector::const_iterator thresholdIterator = thresholdVector.cbegin();
    uint32 numBins = thresholdVector.getNumElements();
    uint32 sparseBinIndex = thresholdVector.getSparseBinIndex();
    bool sparse = sparseBinIndex < numBins;

    // Create a new, empty subset of the statistics...
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = statistics.createSubset(labelIndices);

    for (auto it = thresholdVector.missing_indices_cbegin(); it != thresholdVector.missing_indices_cend(); it++) {
        uint32 i = *it;
        statisticsSubsetPtr->addToMissing(i);
    }

    // In the following, we start by processing the bins in range [0, sparseBinIndex)...
    uint32 numCovered = 0;
    int64 firstR = 0;
    int64 r;

    // Traverse bins in ascending order until the first bin with non-zero weight is encountered...
    for (r = 0; r < sparseBinIndex; r++) {
        uint32 weight = statistics.getBinWeight(r);

        if (weight > 0) {
            // Add the bin to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr->addToSubset(r);
            numCovered += weight;
            break;
        }
    }

    uint32 numAccumulated = numCovered;

    // Traverse the remaining bins in ascending order...
    if (numCovered > 0) {
        for (r = r + 1; r < sparseBinIndex; r++) {
            uint32 weight = statistics.getBinWeight(r);

            // Do only consider bins that are not empty...
            if (weight > 0) {
                // Check if a condition that uses the <= operator (or the == operator in case of a nominal feature)
                // covers at least `minCoverage` examples...
                if (numCovered >= minCoverage) {
                    // Determine the best prediction for the covered examples...
                    const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScores();

                    // Check if the quality of the prediction is better than the quality of the current rule...
                    if (comparator.isImprovement(scoreVector)) {
                        refinement.start = firstR;
                        refinement.end = r;
                        refinement.numCovered = numCovered;
                        refinement.covered = true;
                        refinement.threshold = thresholdIterator[r - 1];
                        refinement.comparator = nominal ? EQ : LEQ;
                        comparator.pushRefinement(refinement, scoreVector);
                    }
                }

                // Check if a condition that uses the > operator (or the != operator in case of a nominal feature)
                // covers at least `minCoverage` examples...
                uint32 coverage = numExamples - numCovered;

                if (coverage >= minCoverage) {
                    // Determine the best prediction for the covered examples...
                    const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScoresUncovered();

                    // Check if the quality of the prediction is better than the quality of the current rule...
                    if (comparator.isImprovement(scoreVector)) {
                        refinement.start = firstR;
                        refinement.end = r;
                        refinement.numCovered = coverage;
                        refinement.covered = false;
                        refinement.threshold = thresholdIterator[r - 1];
                        refinement.comparator = nominal ? NEQ : GR;
                        comparator.pushRefinement(refinement, scoreVector);
                    }
                }

                // Reset the subset in case of a nominal feature, as the previous bins will not be covered by the next
                // condition...
                if (nominal) {
                    statisticsSubsetPtr->resetSubset();
                    numCovered = 0;
                    firstR = r;
                }

                // Add the bin to the subset to mark it as covered by upcoming refinements...
                statisticsSubsetPtr->addToSubset(r);
                numCovered += weight;
                numAccumulated += weight;
            }
        }

        // If any bins have been processed so far and if there is a sparse bin, we must evaluate additional conditions
        // that separate the bins that have been iterated from the remaining ones (including the sparse bin)...
        if (numCovered > 0 && sparse) {
            // Check if a condition that uses the <= operator (or the == operator in case of a nominal feature) covers
            // at least `minCoverage` examples...
            if (numCovered >= minCoverage) {
                // Determine the best prediction for the covered examples...
                const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScores();

                // Check if the quality of the prediction is better than the quality of the current rule...
                if (comparator.isImprovement(scoreVector)) {
                    refinement.start = firstR;
                    refinement.end = sparseBinIndex;
                    refinement.numCovered = numCovered;
                    refinement.covered = true;
                    refinement.threshold = thresholdIterator[sparseBinIndex - 1];
                    refinement.comparator = nominal ? EQ : LEQ;
                    comparator.pushRefinement(refinement, scoreVector);
                }
            }

            // Check if a condition that uses the > operator (or the != operator in case of a nominal feature) covers at
            // least `minCoverage` examples...
            uint32 coverage = numExamples - numCovered;

            if (coverage >= minCoverage) {
                // Determine the best prediction for the covered examples...
                const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScoresUncovered();

                // Check if the quality of the prediction is better than the quality of the current rule...
                if (comparator.isImprovement(scoreVector)) {
                    refinement.start = firstR;
                    refinement.end = sparseBinIndex;
                    refinement.numCovered = coverage;
                    refinement.covered = false;
                    refinement.threshold = thresholdIterator[sparseBinIndex - 1];
                    refinement.comparator = nominal ? NEQ : GR;
                    comparator.pushRefinement(refinement, scoreVector);
                }
            }
        }

        // Reset the subset, if any bins have been processed...
        statisticsSubsetPtr->resetSubset();
    }

    uint32 numAccumulatedPrevious = numAccumulated;

    // We continue by processing the bins in range (sparseBinIndex, numBins)...
    numCovered = 0;
    firstR = ((int64) numBins) - 1;

    // Traverse bins in descending order until the first bin with non-zero weight is encountered...
    for (r = firstR; r > sparseBinIndex; r--) {
        uint32 weight = statistics.getBinWeight(r);

        if (weight > 0) {
            // Add the bin to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr->addToSubset(r);
            numCovered += weight;
            break;
        }
    }

    numAccumulated = numCovered;

    // Traverse the remaining bins in descending order...
    if (numCovered > 0) {
        for (r = r - 1; r > sparseBinIndex; r--) {
            uint32 weight = statistics.getBinWeight(r);

            // Do only consider bins that are not empty...
            if (weight > 0) {
                // Check if a condition that uses the > operator (or the == operator in case of a nominal feature)
                // covers at least `minCoverage` examples...
                if (numCovered >= minCoverage) {
                    // Determine the best prediction for the covered examples...
                    const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScores();

                    // Check if the quality of the prediction is better than the quality of the current rule...
                    if (comparator.isImprovement(scoreVector)) {
                        refinement.start = firstR;
                        refinement.end = r;
                        refinement.numCovered = numCovered;
                        refinement.covered = true;

                        if (nominal) {
                            refinement.threshold = thresholdIterator[firstR];
                            refinement.comparator = EQ;
                        } else {
                            refinement.threshold = thresholdIterator[r];
                            refinement.comparator = GR;
                        }

                        comparator.pushRefinement(refinement, scoreVector);
                    }
                }

                // Check if a condition that uses the <= operator (or the != operator in case of a nominal feature)
                // covers at least `minCoverage` examples...
                uint32 coverage = numExamples - numCovered;

                if (coverage >= minCoverage) {
                    // Determine the best prediction for the covered examples...
                    const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScoresUncovered();

                    // Check if the quality of the prediction is better than the quality of the current rule...
                    if (comparator.isImprovement(scoreVector)) {
                        refinement.start = firstR;
                        refinement.end = r;
                        refinement.numCovered = coverage;
                        refinement.covered = false;

                        if (nominal) {
                            refinement.threshold = thresholdIterator[firstR];
                            refinement.comparator = NEQ;
                        } else {
                            refinement.threshold = thresholdIterator[r];
                            refinement.comparator = LEQ;
                        }

                        comparator.pushRefinement(refinement, scoreVector);
                    }
                }

                // Reset the subset in case of a nominal feature, as the previous bins will not be covered by the next
                // condition...
                if (nominal) {
                    statisticsSubsetPtr->resetSubset();
                    numCovered = 0;
                    firstR = r;
                }

                // Add the bin to the subset to mark it as covered by upcoming refinements...
                statisticsSubsetPtr->addToSubset(r);
                numCovered += weight;
                numAccumulated += weight;
            }
        }

        // If there is a sparse bin, we must evaluate additional conditions that separate the bins in range
        // (sparseBinIndex, numBins) from the remaining ones...
        if (sparse) {
            // Check if a condition that uses the > operator (or the == operator in case of a nominal feature) covers at
            // least `minCoverage` examples...
            if (numCovered >= minCoverage) {
                // Determine the best prediction for the covered examples...
                const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScores();

                // Check if the quality of the prediction is better than the quality of the current rule...
                if (comparator.isImprovement(scoreVector)) {
                    refinement.start = firstR;
                    refinement.end = sparseBinIndex;
                    refinement.numCovered = numCovered;
                    refinement.covered = true;

                    if (nominal) {
                        refinement.threshold = thresholdIterator[firstR];
                        refinement.comparator = EQ;
                    } else {
                        refinement.threshold = thresholdIterator[sparseBinIndex];
                        refinement.comparator = GR;
                    }

                    comparator.pushRefinement(refinement, scoreVector);
                }
            }

            // Check if a condition that uses the <= operator (or the != operator in case of a nominal feature) covers
            // at least `minCoverage` examples...
            uint32 coverage = numExamples - numCovered;

            if (coverage >= minCoverage) {
                // Determine the best prediction for the covered examples...
                const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScoresUncovered();

                // Check if the quality of the prediction is better than the quality of the current rule...
                if (comparator.isImprovement(scoreVector)) {
                    refinement.start = firstR;
                    refinement.end = sparseBinIndex;
                    refinement.numCovered = coverage;
                    refinement.covered = false;

                    if (nominal) {
                        refinement.threshold = thresholdIterator[firstR];
                        refinement.comparator = NEQ;
                    } else {
                        refinement.threshold = thresholdIterator[sparseBinIndex];
                        refinement.comparator = LEQ;
                    }

                    comparator.pushRefinement(refinement, scoreVector);
                }
            }

            // If the feature is nominal and if any bins in the range [0, sparseBinIndex) have been processed earlier,
            // we must test additional conditions that separate the sparse bin from the remaining bins...
            if (nominal && numAccumulatedPrevious > 0) {
                // Reset the subset once again to ensure that the accumulated state includes all bins that have been
                // processed so far...
                statisticsSubsetPtr->resetSubset();

                // Check if the condition `f != thresholdIterator[sparseBinIndex]` covers at least `minCoverage`
                // examples...
                uint32 coverage = numExamples - numAccumulated - numAccumulatedPrevious;

                if (coverage >= minCoverage) {
                    // Determine the best prediction for the covered examples...
                    const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScoresAccumulated();

                    // Check if the quality of the prediction is better than the quality of the current rule...
                    if (comparator.isImprovement(scoreVector)) {
                        refinement.start = sparseBinIndex;
                        refinement.end = sparseBinIndex + 1;
                        refinement.numCovered = coverage;
                        refinement.covered = false;
                        refinement.threshold = thresholdIterator[sparseBinIndex];
                        refinement.comparator = NEQ;
                        comparator.pushRefinement(refinement, scoreVector);
                    }
                }

                // Check if the condition `f == thresholdIterator[sparseBinIndex]` covers at least `minCoverage`
                // examples...
                coverage = numAccumulated + numAccumulatedPrevious;

                if (coverage >= minCoverage) {
                    // Determine the best prediction for the covered examples...
                    const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScoresUncoveredAccumulated();

                    // Check if the quality of the prediction is better than the quality of the current rule...
                    if (comparator.isImprovement(scoreVector)) {
                        refinement.start = sparseBinIndex;
                        refinement.end = sparseBinIndex + 1;
                        refinement.numCovered = coverage;
                        refinement.covered = true;
                        refinement.threshold = thresholdIterator[sparseBinIndex];
                        refinement.comparator = EQ;
                        comparator.pushRefinement(refinement, scoreVector);
                    }
                }
            }
        }
    }
}

template<typename IndexVector>
ApproximateRuleRefinement<IndexVector>::ApproximateRuleRefinement(const IndexVector& labelIndices, uint32 numExamples,
                                                                  uint32 featureIndex, bool nominal,
                                                                  std::unique_ptr<Callback> callbackPtr)
    : labelIndices_(labelIndices), numExamples_(numExamples), featureIndex_(featureIndex), nominal_(nominal),
      callbackPtr_(std::move(callbackPtr)) {}

template<typename IndexVector>
void ApproximateRuleRefinement<IndexVector>::findRefinement(SingleRefinementComparator& comparator,
                                                            uint32 minCoverage) {
    findRefinementInternally(labelIndices_, numExamples_, featureIndex_, nominal_, minCoverage, *callbackPtr_,
                             comparator);
}

template<typename IndexVector>
void ApproximateRuleRefinement<IndexVector>::findRefinement(FixedRefinementComparator& comparator, uint32 minCoverage) {
    findRefinementInternally(labelIndices_, numExamples_, featureIndex_, nominal_, minCoverage, *callbackPtr_,
                             comparator);
}

template class ApproximateRuleRefinement<CompleteIndexVector>;
template class ApproximateRuleRefinement<PartialIndexVector>;
