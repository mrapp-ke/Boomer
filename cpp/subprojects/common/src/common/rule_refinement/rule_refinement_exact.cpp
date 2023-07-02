#include "common/rule_refinement/rule_refinement_exact.hpp"

#include "common/math/math.hpp"

static inline uint32 upperBound(FeatureVector::const_iterator iterator, uint32 start, uint32 end, float32 threshold) {
    while (start < end) {
        uint32 pivot = start + ((end - start) / 2);
        float32 featureValue = iterator[pivot].value;

        if (featureValue <= threshold) {
            start = pivot + 1;
        } else {
            end = pivot;
        }
    }

    return start;
}

static inline void adjustRefinement(Refinement& refinement, FeatureVector::const_iterator iterator) {
    int64 previous = refinement.previous;
    int64 end = refinement.end;

    if (std::abs(previous - end) > 1) {
        if (end < previous) {
            refinement.end = ((int64) upperBound(iterator, end + 1, previous, refinement.threshold)) - 1;
        } else {
            refinement.end = upperBound(iterator, previous + 1, end, refinement.threshold);
        }
    }
}

template<typename IndexIterator, typename RefinementComparator>
static inline void findRefinementInternally(
  const IndexIterator& labelIndices, uint32 numExamples, uint32 featureIndex, bool nominal, uint32 minCoverage,
  bool hasZeroWeights, IRuleRefinementCallback<IImmutableWeightedStatistics, FeatureVector>& callback,
  RefinementComparator& comparator) {
    Refinement refinement;
    refinement.featureIndex = featureIndex;

    // Invoke the callback...
    IRuleRefinementCallback<IImmutableWeightedStatistics, FeatureVector>::Result callbackResult = callback.get();
    const IImmutableWeightedStatistics& statistics = callbackResult.statistics;
    const FeatureVector& featureVector = callbackResult.vector;
    FeatureVector::const_iterator featureVectorIterator = featureVector.cbegin();
    uint32 numFeatureValues = featureVector.getNumElements();

    // Create a new, empty subset of the statistics...
    std::unique_ptr<IWeightedStatisticsSubset> statisticsSubsetPtr = statistics.createSubset(labelIndices);

    for (auto it = featureVector.missing_indices_cbegin(); it != featureVector.missing_indices_cend(); it++) {
        uint32 i = *it;
        statisticsSubsetPtr->addToMissing(i);
    }

    // In the following, we start by processing all examples with feature values < 0...
    uint32 numCovered = 0;
    int64 firstR = 0;
    int64 lastNegativeR = -1;
    float32 previousThreshold = 0;
    int64 previousR = 0;
    int64 r;

    // Traverse examples with feature values < 0 in ascending order until the first example with non-zero weight is
    // encountered...
    for (r = 0; r < numFeatureValues; r++) {
        float32 currentThreshold = featureVectorIterator[r].value;

        if (currentThreshold >= 0) {
            break;
        }

        lastNegativeR = r;
        uint32 i = featureVectorIterator[r].index;

        if (statisticsSubsetPtr->hasNonZeroWeight(i)) {
            // Add the example to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr->addToSubset(i);
            numCovered++;
            previousThreshold = currentThreshold;
            previousR = r;
            break;
        }
    }

    uint32 numAccumulated = numCovered;

    // Traverse the remaining examples with feature values < 0 in ascending order...
    if (numCovered > 0) {
        for (r = r + 1; r < numFeatureValues; r++) {
            float32 currentThreshold = featureVectorIterator[r].value;

            if (currentThreshold >= 0) {
                break;
            }

            lastNegativeR = r;
            uint32 i = featureVectorIterator[r].index;

            // Do only consider examples that are included in the current sub-sample...
            if (statisticsSubsetPtr->hasNonZeroWeight(i)) {
                // Thresholds that separate between examples with the same feature value must not be considered...
                if (previousThreshold != currentThreshold) {
                    // Check if a condition that uses the <= operator (or the == operator in case of a nominal feature)
                    // covers at least `minCoverage` examples...
                    if (numCovered >= minCoverage) {
                        // Determine the best prediction for the covered examples...
                        const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScores();

                        // Check if the quality of the prediction is better than the quality of the current rule...
                        if (comparator.isImprovement(scoreVector)) {
                            refinement.start = firstR;
                            refinement.end = r;
                            refinement.previous = previousR;
                            refinement.numCovered = numCovered;
                            refinement.covered = true;

                            if (nominal) {
                                refinement.comparator = EQ;
                                refinement.threshold = previousThreshold;
                            } else {
                                refinement.comparator = LEQ;
                                refinement.threshold = arithmeticMean(previousThreshold, currentThreshold);
                            }

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
                            refinement.previous = previousR;
                            refinement.numCovered = coverage;
                            refinement.covered = false;

                            if (nominal) {
                                refinement.comparator = NEQ;
                                refinement.threshold = previousThreshold;
                            } else {
                                refinement.comparator = GR;
                                refinement.threshold = arithmeticMean(previousThreshold, currentThreshold);
                            }

                            comparator.pushRefinement(refinement, scoreVector);
                        }
                    }

                    // Reset the subset in case of a nominal feature, as the previous examples will not be covered by
                    // the next condition...
                    if (nominal) {
                        statisticsSubsetPtr->resetSubset();
                        numCovered = 0;
                        firstR = r;
                    }
                }

                previousThreshold = currentThreshold;
                previousR = r;

                // Add the example to the subset to mark it as covered by upcoming refinements...
                statisticsSubsetPtr->addToSubset(i);
                numCovered++;
                numAccumulated++;
            }
        }

        // If the feature is nominal and the examples that have been iterated so far do not have the same feature value,
        // or if not all examples have been iterated so far, we must evaluate additional conditions
        // `f == previousThreshold` and `f != previousThreshold`...
        if (nominal && numCovered > 0 && (numCovered < numAccumulated || numAccumulated < numExamples)) {
            // Check if a condition that uses the == operator covers at least `minCoverage` examples...
            if (numCovered >= minCoverage) {
                // Determine the best prediction for the covered examples...
                const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScores();

                // Check if the quality of the prediction is better than the quality of the current rule...
                if (comparator.isImprovement(scoreVector)) {
                    refinement.start = firstR;
                    refinement.end = (lastNegativeR + 1);
                    refinement.previous = previousR;
                    refinement.numCovered = numCovered;
                    refinement.covered = true;
                    refinement.comparator = EQ;
                    refinement.threshold = previousThreshold;
                    comparator.pushRefinement(refinement, scoreVector);
                }
            }

            // Check if a condition that uses the != operator covers at least `minCoverage` examples...
            uint32 coverage = numExamples - numCovered;

            if (coverage >= minCoverage) {
                // Determine the best prediction for the covered examples...
                const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScoresUncovered();

                // Check if the quality of the prediction is better than the quality of the current rule...
                if (comparator.isImprovement(scoreVector)) {
                    refinement.start = firstR;
                    refinement.end = (lastNegativeR + 1);
                    refinement.previous = previousR;
                    refinement.numCovered = coverage;
                    refinement.covered = false;
                    refinement.comparator = NEQ;
                    refinement.threshold = previousThreshold;
                    comparator.pushRefinement(refinement, scoreVector);
                }
            }
        }

        // Reset the subset, if any examples with feature value < 0 have been processed...
        statisticsSubsetPtr->resetSubset();
    }

    float32 previousThresholdNegative = previousThreshold;
    int64 previousRNegative = previousR;
    uint32 numAccumulatedNegative = numAccumulated;

    // We continue by processing all examples with feature values >= 0...
    numCovered = 0;
    firstR = ((int64) numFeatureValues) - 1;

    // Traverse examples with feature values >= 0 in descending order until the first example with non-zero weight is
    // encountered...
    for (r = firstR; r > lastNegativeR; r--) {
        uint32 i = featureVectorIterator[r].index;

        if (statisticsSubsetPtr->hasNonZeroWeight(i)) {
            // Add the example to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr->addToSubset(i);
            numCovered++;
            previousThreshold = featureVectorIterator[r].value;
            previousR = r;
            break;
        }
    }

    numAccumulated = numCovered;

    // Traverse the remaining examples with feature values >= 0 in descending order...
    if (numCovered > 0) {
        for (r = r - 1; r > lastNegativeR; r--) {
            uint32 i = featureVectorIterator[r].index;

            // Do only consider examples that are included in the current sub-sample...
            if (statisticsSubsetPtr->hasNonZeroWeight(i)) {
                float32 currentThreshold = featureVectorIterator[r].value;

                // Thresholds that separate between examples with the same feature value must not be considered...
                if (previousThreshold != currentThreshold) {
                    // Check if a condition that uses the > operator (or the == operator in case of a nominal feature)
                    // covers at least `minCoverage` examples...
                    if (numCovered >= minCoverage) {
                        // Determine the best prediction for the covered examples...
                        const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScores();

                        // Check if the quality of the prediction is better than the quality of the current rule...
                        if (comparator.isImprovement(scoreVector)) {
                            refinement.start = firstR;
                            refinement.end = r;
                            refinement.previous = previousR;
                            refinement.numCovered = numCovered;
                            refinement.covered = true;

                            if (nominal) {
                                refinement.comparator = EQ;
                                refinement.threshold = previousThreshold;
                            } else {
                                refinement.comparator = GR;
                                refinement.threshold = arithmeticMean(currentThreshold, previousThreshold);
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
                            refinement.previous = previousR;
                            refinement.numCovered = coverage;
                            refinement.covered = false;

                            if (nominal) {
                                refinement.comparator = NEQ;
                                refinement.threshold = previousThreshold;
                            } else {
                                refinement.comparator = LEQ;
                                refinement.threshold = arithmeticMean(currentThreshold, previousThreshold);
                            }

                            comparator.pushRefinement(refinement, scoreVector);
                        }
                    }

                    // Reset the subset in case of a nominal feature, as the previous examples will not be covered by
                    // the next condition...
                    if (nominal) {
                        statisticsSubsetPtr->resetSubset();
                        numCovered = 0;
                        firstR = r;
                    }
                }

                previousThreshold = currentThreshold;
                previousR = r;

                // Add the example to the subset to mark it as covered by upcoming refinements...
                statisticsSubsetPtr->addToSubset(i);
                numCovered++;
                numAccumulated++;
            }
        }
    }

    // If the feature is nominal and the examples with feature values >= 0 that have been iterated so far do not all
    // have the same feature value, we must evaluate additional conditions `f == previousThreshold` and
    // `f != previousThreshold`...
    if (nominal && numCovered > 0 && numCovered < numAccumulated) {
        // Check if a condition that uses the == operator covers at least `minCoverage` examples...
        if (numCovered >= minCoverage) {
            // Determine the best prediction for the covered examples...
            const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScores();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = firstR;
                refinement.end = lastNegativeR;
                refinement.previous = previousR;
                refinement.numCovered = numCovered;
                refinement.covered = true;
                refinement.comparator = EQ;
                refinement.threshold = previousThreshold;
                comparator.pushRefinement(refinement, scoreVector);
            }
        }

        // Check if a condition that uses the != operator covers at least `minCoverage` examples...
        uint32 coverage = numExamples - numCovered;

        if (coverage >= minCoverage) {
            // Determine the best prediction for the covered examples...
            const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScoresUncovered();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = firstR;
                refinement.end = lastNegativeR;
                refinement.previous = previousR;
                refinement.numCovered = coverage;
                refinement.covered = false;
                refinement.comparator = NEQ;
                refinement.threshold = previousThreshold;
                comparator.pushRefinement(refinement, scoreVector);
            }
        }
    }

    uint32 numAccumulatedTotal = numAccumulatedNegative + numAccumulated;

    // If the number of all examples that have been iterated so far (including those with feature values < 0 and those
    // with feature values >= 0) is less than the total number of examples, this means that there are examples with
    // sparse, i.e. zero, feature values. In such case, we must explicitly test conditions that separate these examples
    // from the ones that have already been iterated...
    if (numAccumulatedTotal > 0 && numAccumulatedTotal < numExamples) {
        // If the feature is nominal, we must reset the subset once again to ensure that the accumulated state includes
        // all examples that have been processed so far...
        if (nominal) {
            statisticsSubsetPtr->resetSubset();
            firstR = ((int64) numFeatureValues) - 1;
        }

        // Check if the condition `f > previousThreshold / 2` (or `f != 0` in case of a nominal feature) covers at least
        // `minCoverage` examples...
        uint32 coverage = nominal ? numAccumulatedTotal : numAccumulated;

        if (coverage >= minCoverage) {
            // Determine the best prediction for the covered examples...
            const IScoreVector& scoreVector =
              nominal ? statisticsSubsetPtr->calculateScoresAccumulated() : statisticsSubsetPtr->calculateScores();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = firstR;
                refinement.covered = true;
                refinement.numCovered = coverage;

                if (nominal) {
                    refinement.end = -1;
                    refinement.previous = -1;
                    refinement.comparator = NEQ;
                    refinement.threshold = 0.0;
                } else {
                    refinement.end = lastNegativeR;
                    refinement.previous = previousR;
                    refinement.comparator = GR;
                    refinement.threshold = previousThreshold * 0.5;
                }

                comparator.pushRefinement(refinement, scoreVector);
            }
        }

        // Check if the condition `f <= previousThreshold / 2` (or `f == 0` in case of a nominal feature) covers at
        // least `minCoverage` examples...
        coverage = numExamples - (nominal ? numAccumulatedTotal : numAccumulated);

        if (coverage >= minCoverage) {
            // Determine the best prediction for the covered examples...
            const IScoreVector& scoreVector = nominal ? statisticsSubsetPtr->calculateScoresUncoveredAccumulated()
                                                      : statisticsSubsetPtr->calculateScoresUncovered();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = firstR;
                refinement.covered = false;
                refinement.numCovered = coverage;

                if (nominal) {
                    refinement.end = -1;
                    refinement.previous = -1;
                    refinement.comparator = EQ;
                    refinement.threshold = 0.0;
                } else {
                    refinement.end = lastNegativeR;
                    refinement.previous = previousR;
                    refinement.numCovered = (numExamples - numAccumulated);
                    refinement.comparator = LEQ;
                    refinement.threshold = previousThreshold * 0.5;
                }

                comparator.pushRefinement(refinement, scoreVector);
            }
        }
    }

    // If the feature is numerical and there are other examples than those with feature values < 0 that have been
    // processed earlier, we must evaluate additional conditions that separate the examples with feature values < 0 from
    // the remaining ones (unlike in the nominal case, these conditions cannot be evaluated earlier, because it remains
    // unclear what the thresholds of the conditions should be until the examples with feature values >= 0 have been
    // processed).
    if (!nominal && numAccumulatedNegative > 0 && numAccumulatedNegative < numExamples) {
        // Check if a condition that uses the <= operator covers at least `minCoverage` examples...
        if (numAccumulatedNegative >= minCoverage) {
            // Determine the best prediction for the covered examples...
            const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScoresAccumulated();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = 0;
                refinement.end = (lastNegativeR + 1);
                refinement.previous = previousRNegative;
                refinement.numCovered = numAccumulatedNegative;
                refinement.covered = true;
                refinement.comparator = LEQ;

                if (numAccumulatedTotal < numExamples) {
                    // If the condition separates an example with feature value < 0 from an (sparse) example with
                    // feature value == 0
                    refinement.threshold = previousThresholdNegative * 0.5;
                } else {
                    // If the condition separates an example with feature value < 0 from an example with feature value
                    // > 0
                    refinement.threshold = arithmeticMean(previousThresholdNegative, previousThreshold);
                }

                comparator.pushRefinement(refinement, scoreVector);
            }
        }

        // Check if a condition that uses the > operator covers at least `minCoverage` examples...
        uint32 coverage = numExamples - numAccumulatedNegative;

        if (coverage >= minCoverage) {
            // Determine the best prediction for the covered examples...
            const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScoresUncoveredAccumulated();

            // Check if the quality of the prediction is better than the quality of the current rule...
            if (comparator.isImprovement(scoreVector)) {
                refinement.start = 0;
                refinement.end = (lastNegativeR + 1);
                refinement.previous = previousRNegative;
                refinement.numCovered = coverage;
                refinement.covered = false;
                refinement.comparator = GR;

                if (numAccumulatedTotal < numExamples) {
                    // If the condition separates an example with feature value < 0 from an (sparse) example with
                    // feature value == 0
                    refinement.threshold = previousThresholdNegative * 0.5;
                } else {
                    // If the condition separates an example with feature value < 0 from an example with feature value
                    // > 0
                    refinement.threshold = arithmeticMean(previousThresholdNegative, previousThreshold);
                }

                comparator.pushRefinement(refinement, scoreVector);
            }
        }
    }

    // If there are examples with zero weights, those examples have not been considered when searching for potential
    // refinements. In this case, we need to identify the examples that are covered by a refinement, including those
    // that have previously been ignored, and adjust the value `refinement.end`, which specifies the position that
    // separates the covered from the uncovered examples, accordingly.
    if (hasZeroWeights) {
        for (auto it = comparator.begin(); it != comparator.end(); it++) {
            adjustRefinement(*it, featureVectorIterator);
        }
    }
}

template<typename IndexVector>
ExactRuleRefinement<IndexVector>::ExactRuleRefinement(const IndexVector& labelIndices, uint32 numExamples,
                                                      uint32 featureIndex, bool nominal, bool hasZeroWeights,
                                                      std::unique_ptr<Callback> callbackPtr)
    : labelIndices_(labelIndices), numExamples_(numExamples), featureIndex_(featureIndex), nominal_(nominal),
      hasZeroWeights_(hasZeroWeights), callbackPtr_(std::move(callbackPtr)) {}

template<typename IndexVector>
void ExactRuleRefinement<IndexVector>::findRefinement(SingleRefinementComparator& comparator, uint32 minCoverage) {
    findRefinementInternally(labelIndices_, numExamples_, featureIndex_, nominal_, minCoverage, hasZeroWeights_,
                             *callbackPtr_, comparator);
}

template<typename IndexVector>
void ExactRuleRefinement<IndexVector>::findRefinement(FixedRefinementComparator& comparator, uint32 minCoverage) {
    findRefinementInternally(labelIndices_, numExamples_, featureIndex_, nominal_, minCoverage, hasZeroWeights_,
                             *callbackPtr_, comparator);
}

template class ExactRuleRefinement<CompleteIndexVector>;
template class ExactRuleRefinement<PartialIndexVector>;
