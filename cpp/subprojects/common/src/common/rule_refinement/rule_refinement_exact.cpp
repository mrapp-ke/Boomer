#include "common/rule_refinement/rule_refinement_exact.hpp"
#include "common/math/math.hpp"


template<class T>
ExactRuleRefinement<T>::ExactRuleRefinement(
        std::unique_ptr<IHeadRefinement> headRefinementPtr, const T& labelIndices, uint32 totalSumOfWeights,
        uint32 featureIndex, bool nominal,
        std::unique_ptr<IRuleRefinementCallback<FeatureVector, IWeightVector>> callbackPtr)
    : headRefinementPtr_(std::move(headRefinementPtr)), labelIndices_(labelIndices),
      totalSumOfWeights_(totalSumOfWeights), featureIndex_(featureIndex), nominal_(nominal),
      callbackPtr_(std::move(callbackPtr)) {

}

template<class T>
void ExactRuleRefinement<T>::findRefinement(const AbstractEvaluatedPrediction* currentHead) {
    std::unique_ptr<Refinement> refinementPtr = std::make_unique<Refinement>();
    refinementPtr->featureIndex = featureIndex_;
    const AbstractEvaluatedPrediction* bestHead = currentHead;

    // Invoke the callback...
    std::unique_ptr<IRuleRefinementCallback<FeatureVector, IWeightVector>::Result> callbackResultPtr =
        callbackPtr_->get();
    const IImmutableStatistics& statistics = callbackResultPtr->statistics_;
    const IWeightVector& weights = callbackResultPtr->weights_;
    const FeatureVector& featureVector = callbackResultPtr->vector_;
    FeatureVector::const_iterator iterator = featureVector.cbegin();
    uint32 numElements = featureVector.getNumElements();

    // Create a new, empty subset of the statistics...
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = labelIndices_.createSubset(statistics);

    for (auto it = featureVector.missing_indices_cbegin(); it != featureVector.missing_indices_cend(); it++) {
        uint32 i = *it;
        uint32 weight = weights.getWeight(i);
        statisticsSubsetPtr->addToMissing(i, weight);
    }

    // In the following, we start by processing all examples with feature values < 0...
    uint32 sumOfWeights = 0;
    intp firstR = 0;
    intp lastNegativeR = -1;
    float32 previousThreshold = 0;
    intp previousR = 0;
    intp r;

    // Traverse examples with feature values < 0 in ascending order until the first example with weight > 0 is
    // encountered...
    for (r = 0; r < numElements; r++) {
        float32 currentThreshold = iterator[r].value;

        if (currentThreshold >= 0) {
            break;
        }

        lastNegativeR = r;
        uint32 i = iterator[r].index;
        uint32 weight = weights.getWeight(i);

        if (weight > 0) {
            // Add the example to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr->addToSubset(i, weight);
            sumOfWeights += weight;
            previousThreshold = currentThreshold;
            previousR = r;
            break;
        }
    }

    uint32 accumulatedSumOfWeights = sumOfWeights;

    // Traverse the remaining examples with feature values < 0 in ascending order...
    if (sumOfWeights > 0) {
        for (r = r + 1; r < numElements; r++) {
            float32 currentThreshold = iterator[r].value;

            if (currentThreshold >= 0) {
                break;
            }

            lastNegativeR = r;
            uint32 i = iterator[r].index;
            uint32 weight = weights.getWeight(i);

            // Do only consider examples that are included in the current sub-sample...
            if (weight > 0) {
                // Split points between examples with the same feature value must not be considered...
                if (previousThreshold != currentThreshold) {
                    // Find and evaluate the best head for the current refinement, if a condition that uses the <=
                    // operator (or the == operator in case of a nominal feature) is used...
                    const AbstractEvaluatedPrediction* head = headRefinementPtr_->findHead(bestHead,
                                                                                           *statisticsSubsetPtr, false,
                                                                                           false);

                    // If the refinement is better than the current rule...
                    if (head != nullptr) {
                        bestHead = head;
                        refinementPtr->start = firstR;
                        refinementPtr->end = r;
                        refinementPtr->previous = previousR;
                        refinementPtr->coveredWeights = sumOfWeights;
                        refinementPtr->covered = true;

                        if (nominal_) {
                            refinementPtr->comparator = EQ;
                            refinementPtr->threshold = previousThreshold;
                        } else {
                            refinementPtr->comparator = LEQ;
                            refinementPtr->threshold = arithmeticMean(previousThreshold, currentThreshold);
                        }
                    }

                    // Find and evaluate the best head for the current refinement, if a condition that uses the >
                    // operator (or the != operator in case of a nominal feature) is used...
                    head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, true, false);

                    // If the refinement is better than the current rule...
                    if (head != nullptr) {
                        bestHead = head;
                        refinementPtr->start = firstR;
                        refinementPtr->end = r;
                        refinementPtr->previous = previousR;
                        refinementPtr->coveredWeights = (totalSumOfWeights_ - sumOfWeights);
                        refinementPtr->covered = false;

                        if (nominal_) {
                            refinementPtr->comparator = NEQ;
                            refinementPtr->threshold = previousThreshold;
                        } else {
                            refinementPtr->comparator = GR;
                            refinementPtr->threshold = arithmeticMean(previousThreshold, currentThreshold);
                        }
                    }

                    // Reset the subset in case of a nominal feature, as the previous examples will not be covered by
                    // the next condition...
                    if (nominal_) {
                        statisticsSubsetPtr->resetSubset();
                        sumOfWeights = 0;
                        firstR = r;
                    }
                }

                previousThreshold = currentThreshold;
                previousR = r;

                // Add the example to the subset to mark it as covered by upcoming refinements...
                statisticsSubsetPtr->addToSubset(i, weight);
                sumOfWeights += weight;
                accumulatedSumOfWeights += weight;
            }
        }

        // If the feature is nominal and the examples that have been iterated so far do not all have the same feature
        // value, or if not all examples have been iterated so far, we must evaluate additional conditions
        // `f == previous_threshold` and `f != previous_threshold`...
        if (nominal_ && sumOfWeights > 0 && (sumOfWeights < accumulatedSumOfWeights
                                             || accumulatedSumOfWeights < totalSumOfWeights_)) {
            // Find and evaluate the best head for the current refinement, if a condition that uses the == operator is
            // used...
            const AbstractEvaluatedPrediction* head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr,
                                                                                   false, false);

            // If the refinement is better than the current rule...
            if (head != nullptr) {
                bestHead = head;
                refinementPtr->start = firstR;
                refinementPtr->end = (lastNegativeR + 1);
                refinementPtr->previous = previousR;
                refinementPtr->coveredWeights = sumOfWeights;
                refinementPtr->covered = true;
                refinementPtr->comparator = EQ;
                refinementPtr->threshold = previousThreshold;
            }

            // Find and evaluate the best head for the current refinement, if a condition that uses the != operator is
            // used...
            head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, true, false);

            // If the refinement is better than the current rule...
            if (head != nullptr) {
                bestHead = head;
                refinementPtr->start = firstR;
                refinementPtr->end = (lastNegativeR + 1);
                refinementPtr->previous = previousR;
                refinementPtr->coveredWeights = (totalSumOfWeights_ - sumOfWeights);
                refinementPtr->covered = false;
                refinementPtr->comparator = NEQ;
                refinementPtr->threshold = previousThreshold;
            }
        }

        // Reset the subset, if any examples with feature value < 0 have been processed...
        statisticsSubsetPtr->resetSubset();
    }

    float32 previousThresholdNegative = previousThreshold;
    intp previousRNegative = previousR;
    uint32 accumulatedSumOfWeightsNegative = accumulatedSumOfWeights;

    // We continue by processing all examples with feature values >= 0...
    sumOfWeights = 0;
    firstR = ((intp) numElements) - 1;

    // Traverse examples with feature values >= 0 in descending order until the first example with weight > 0 is
    // encountered...
    for (r = firstR; r > lastNegativeR; r--) {
        uint32 i = iterator[r].index;
        uint32 weight = weights.getWeight(i);

        if (weight > 0) {
            // Add the example to the subset to mark it as covered by upcoming refinements...
            statisticsSubsetPtr->addToSubset(i, weight);
            sumOfWeights += weight;
            previousThreshold = iterator[r].value;
            previousR = r;
            break;
        }
    }

    accumulatedSumOfWeights = sumOfWeights;

    // Traverse the remaining examples with feature values >= 0 in descending order...
    if (sumOfWeights > 0) {
        for (r = r - 1; r > lastNegativeR; r--) {
            uint32 i = iterator[r].index;
            uint32 weight = weights.getWeight(i);

            // Do only consider examples that are included in the current sub-sample...
            if (weight > 0) {
                float32 currentThreshold = iterator[r].value;

                // Split points between examples with the same feature value must not be considered...
                if (previousThreshold != currentThreshold) {
                    // Find and evaluate the best head for the current refinement, if a condition that uses the
                    // > operator (or the == operator in case of a nominal feature) is used...
                    const AbstractEvaluatedPrediction* head = headRefinementPtr_->findHead(bestHead,
                                                                                           *statisticsSubsetPtr, false,
                                                                                           false);

                    // If the refinement is better than the current rule...
                    if (head != nullptr) {
                        bestHead = head;
                        refinementPtr->start = firstR;
                        refinementPtr->end = r;
                        refinementPtr->previous = previousR;
                        refinementPtr->coveredWeights = sumOfWeights;
                        refinementPtr->covered = true;

                        if (nominal_) {
                            refinementPtr->comparator = EQ;
                            refinementPtr->threshold = previousThreshold;
                        } else {
                            refinementPtr->comparator = GR;
                            refinementPtr->threshold = arithmeticMean(currentThreshold, previousThreshold);
                        }
                    }

                    // Find and evaluate the best head for the current refinement, if a condition that uses the <=
                    // operator (or the != operator in case of a nominal feature) is used...
                    head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, true, false);

                    // If the refinement is better than the current rule...
                    if (head != nullptr) {
                        bestHead = head;
                        refinementPtr->start = firstR;
                        refinementPtr->end = r;
                        refinementPtr->previous = previousR;
                        refinementPtr->coveredWeights = (totalSumOfWeights_ - sumOfWeights);
                        refinementPtr->covered = false;

                        if (nominal_) {
                            refinementPtr->comparator = NEQ;
                            refinementPtr->threshold = previousThreshold;
                        } else {
                            refinementPtr->comparator = LEQ;
                            refinementPtr->threshold = arithmeticMean(currentThreshold, previousThreshold);
                        }
                    }

                    // Reset the subset in case of a nominal feature, as the previous examples will not be covered by
                    // the next condition...
                    if (nominal_) {
                        statisticsSubsetPtr->resetSubset();
                        sumOfWeights = 0;
                        firstR = r;
                    }
                }

                previousThreshold = currentThreshold;
                previousR = r;

                // Add the example to the subset to mark it as covered by upcoming refinements...
                statisticsSubsetPtr->addToSubset(i, weight);
                sumOfWeights += weight;
                accumulatedSumOfWeights += weight;
            }
        }
    }

    // If the feature is nominal and the examples with feature values >= 0 that have been iterated so far do not all
    // have the same feature value, we must evaluate additional conditions `f == previous_threshold` and
    // `f != previous_threshold`...
    if (nominal_ && sumOfWeights > 0 && sumOfWeights < accumulatedSumOfWeights) {
        // Find and evaluate the best head for the current refinement, if a condition that uses the == operator is
        // used...
        const AbstractEvaluatedPrediction* head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, false,
                                                                               false);

        // If the refinement is better than the current rule...
        if (head != nullptr) {
            bestHead = head;
            refinementPtr->start = firstR;
            refinementPtr->end = lastNegativeR;
            refinementPtr->previous = previousR;
            refinementPtr->coveredWeights = sumOfWeights;
            refinementPtr->covered = true;
            refinementPtr->comparator = EQ;
            refinementPtr->threshold = previousThreshold;
        }

        // Find and evaluate the best head for the current refinement, if a condition that uses the != operator is
        // used...
        head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, true, false);

        // If the refinement is better than the current rule...
        if (head != nullptr) {
            bestHead = head;
            refinementPtr->start = firstR;
            refinementPtr->end = lastNegativeR;
            refinementPtr->previous = previousR;
            refinementPtr->coveredWeights = (totalSumOfWeights_ - sumOfWeights);
            refinementPtr->covered = false;
            refinementPtr->comparator = NEQ;
            refinementPtr->threshold = previousThreshold;
        }
    }

    uint32 totalAccumulatedSumOfWeights = accumulatedSumOfWeightsNegative + accumulatedSumOfWeights;

    // If the sum of weights of all examples that have been iterated so far (including those with feature values < 0 and
    // those with feature values >= 0) is less than the sum of weights of all examples, this means that there are
    // examples with sparse, i.e. zero, feature values. In such case, we must explicitly test conditions that separate
    // these examples from the ones that have already been iterated...
    if (totalAccumulatedSumOfWeights > 0 && totalAccumulatedSumOfWeights < totalSumOfWeights_) {
        // If the feature is nominal, we must reset the subset once again to ensure that the accumulated state includes
        // all examples that have been processed so far...
        if (nominal_) {
            statisticsSubsetPtr->resetSubset();
            firstR = ((intp) numElements) - 1;
        }

        // Find and evaluate the best head for the current refinement, if the condition `f > previous_threshold / 2` (or
        // the condition `f != 0` in case of a nominal feature) is used...
        const AbstractEvaluatedPrediction* head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, false,
                                                                               nominal_);

        // If the refinement is better than the current rule...
        if (head != nullptr) {
            bestHead = head;
            refinementPtr->start = firstR;
            refinementPtr->covered = true;

            if (nominal_) {
                refinementPtr->end = -1;
                refinementPtr->previous = -1;
                refinementPtr->coveredWeights = totalAccumulatedSumOfWeights;
                refinementPtr->comparator = NEQ;
                refinementPtr->threshold = 0.0;
            } else {
                refinementPtr->end = lastNegativeR;
                refinementPtr->previous = previousR;
                refinementPtr->coveredWeights = accumulatedSumOfWeights;
                refinementPtr->comparator = GR;
                refinementPtr->threshold = previousThreshold * 0.5;
            }
        }

        // Find and evaluate the best head for the current refinement, if the condition `f <= previous_threshold / 2`
        // (or `f == 0` in case of a nominal feature) is used...
        head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, true, nominal_);

        // If the refinement is better than the current rule...
        if (head != nullptr) {
            bestHead = head;
            refinementPtr->start = firstR;
            refinementPtr->covered = false;

            if (nominal_) {
                refinementPtr->end = -1;
                refinementPtr->previous = -1;
                refinementPtr->coveredWeights = (totalSumOfWeights_ - totalAccumulatedSumOfWeights);
                refinementPtr->comparator = EQ;
                refinementPtr->threshold = 0.0;
            } else {
                refinementPtr->end = lastNegativeR;
                refinementPtr->previous = previousR;
                refinementPtr->coveredWeights = (totalSumOfWeights_ - accumulatedSumOfWeights);
                refinementPtr->comparator = LEQ;
                refinementPtr->threshold = previousThreshold * 0.5;
            }
        }
    }

    // If the feature is numerical and there are other examples than those with feature values < 0 that have been
    // processed earlier, we must evaluate additional conditions that separate the examples with feature values < 0 from
    // the remaining ones (unlike in the nominal case, these conditions cannot be evaluated earlier, because it remains
    // unclear what the thresholds of the conditions should be until the examples with feature values >= 0 have been
    // processed).
    if (!nominal_ && accumulatedSumOfWeightsNegative > 0 && accumulatedSumOfWeightsNegative < totalSumOfWeights_) {
        // Find and evaluate the best head for the current refinement, if the condition that uses the <= operator is
        // used...
        const AbstractEvaluatedPrediction* head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, false,
                                                                               true);

        // If the refinement is better than the current rule...
        if (head != nullptr) {
            bestHead = head;
            refinementPtr->start = 0;
            refinementPtr->end = (lastNegativeR + 1);
            refinementPtr->previous = previousRNegative;
            refinementPtr->coveredWeights = accumulatedSumOfWeightsNegative;
            refinementPtr->covered = true;
            refinementPtr->comparator = LEQ;

            if (totalAccumulatedSumOfWeights < totalSumOfWeights_) {
                // If the condition separates an example with feature value < 0 from an (sparse) example with feature
                // value == 0
                refinementPtr->threshold = previousThresholdNegative * 0.5;
            } else {
                // If the condition separates an example with feature value < 0 from an example with feature value > 0
                refinementPtr->threshold = arithmeticMean(previousThresholdNegative, previousThreshold);
            }
        }

        // Find and evaluate the best head for the current refinement, if the condition that uses the > operator is
        // used...
        head = headRefinementPtr_->findHead(bestHead, *statisticsSubsetPtr, true, true);

        // If the refinement is better than the current rule...
        if (head != nullptr) {
            bestHead = head;
            refinementPtr->start = 0;
            refinementPtr->end = (lastNegativeR + 1);
            refinementPtr->previous = previousRNegative;
            refinementPtr->coveredWeights = (totalSumOfWeights_ - accumulatedSumOfWeightsNegative);
            refinementPtr->covered = false;
            refinementPtr->comparator = GR;

            if (totalAccumulatedSumOfWeights < totalSumOfWeights_) {
                // If the condition separates an example with feature value < 0 from an (sparse) example with feature
                // value == 0
                refinementPtr->threshold = previousThresholdNegative * 0.5;
            } else {
                // If the condition separates an example with feature value < 0 from an example with feature value > 0
                refinementPtr->threshold = arithmeticMean(previousThresholdNegative, previousThreshold);
            }
        }
    }

    refinementPtr->headPtr = headRefinementPtr_->pollHead();
    refinementPtr_ = std::move(refinementPtr);
}

template<class T>
std::unique_ptr<Refinement> ExactRuleRefinement<T>::pollRefinement() {
    return std::move(refinementPtr_);
}

template class ExactRuleRefinement<FullIndexVector>;
template class ExactRuleRefinement<PartialIndexVector>;
