#include "common/rule_induction/rule_induction_top_down_beam_search.hpp"

#include "common/math/math.hpp"
#include "common/util/validation.hpp"
#include "rule_induction_common.hpp"
#include "rule_induction_top_down_common.hpp"

#include <algorithm>

/**
 * A single entry of a beam, corresponding to a rule that may be further refined. It stores the conditions and the head
 * of the current rule, as well as an object of type `IThresholdsSubset` that is required to search for potential
 * refinements of the rule and an `IIndexVector` that provides access to the indices of the labels for which these
 * refinements may predict.
 */
struct BeamEntry final {
    public:

        /**
         * An unique pointer to an object of type `ConditionList` that stores the conditions of the rule.
         */
        std::unique_ptr<ConditionList> conditionListPtr;

        /**
         * An unique pointer to an object of type `AbstractEvaluatedPrediction` that stores the prediction of the rule,
         * as well as its quality.
         */
        std::unique_ptr<AbstractEvaluatedPrediction> headPtr;

        /**
         * An unique pointer to an object of type `IThresholdsSubset` that may be used to search for potential
         * refinements of the rule.
         */
        std::unique_ptr<IThresholdsSubset> thresholdsSubsetPtr;

        /**
         * A pointer to an object of type `IIndexVector` that provides access to the indices of the labels  for which
         * potential refinements of the rule may predict.
         */
        const IIndexVector* labelIndices;
};

static inline void initializeEntry(BeamEntry& entry, Refinement& refinement,
                                   std::unique_ptr<IThresholdsSubset> thresholdsSubsetPtr,
                                   const IIndexVector& labelIndices, bool keepHead) {
    thresholdsSubsetPtr->filterThresholds(refinement);
    entry.thresholdsSubsetPtr = std::move(thresholdsSubsetPtr);
    entry.conditionListPtr = std::make_unique<ConditionList>();
    entry.conditionListPtr->addCondition(refinement);
    entry.headPtr = std::move(refinement.headPtr);
    entry.labelIndices = keepHead ? entry.headPtr.get() : &labelIndices;
}

static inline void copyEntry(BeamEntry& newEntry, BeamEntry& oldEntry, Refinement& refinement,
                             std::unique_ptr<IThresholdsSubset> thresholdsSubsetPtr,
                             std::unique_ptr<ConditionList> conditionListPtr, bool keepHead, uint32 minCoverage) {
    thresholdsSubsetPtr->filterThresholds(refinement);
    newEntry.thresholdsSubsetPtr = std::move(thresholdsSubsetPtr);
    newEntry.conditionListPtr = std::move(conditionListPtr);
    newEntry.conditionListPtr->addCondition(refinement);
    newEntry.headPtr = std::move(refinement.headPtr);

    if (refinement.numCovered <= minCoverage) {
        newEntry.labelIndices = nullptr;
    } else {
        newEntry.labelIndices = keepHead ? newEntry.headPtr.get() : oldEntry.labelIndices;
    }
}

static inline void copyEntry(BeamEntry& newEntry, BeamEntry& oldEntry) {
    newEntry.thresholdsSubsetPtr = std::move(oldEntry.thresholdsSubsetPtr);
    newEntry.conditionListPtr = std::move(oldEntry.conditionListPtr);
    newEntry.headPtr = std::move(oldEntry.headPtr);
    newEntry.labelIndices = nullptr;
}

static inline const Quality& updateOrder(RuleCompareFunction ruleCompareFunction,
                                         std::vector<std::reference_wrapper<BeamEntry>>& order) {
    std::sort(order.begin(), order.end(), [=](const BeamEntry& a, const BeamEntry& b) {
        return ruleCompareFunction.compare(*a.headPtr, *b.headPtr);
    });
    const BeamEntry& worstEntry = order.back();
    return *worstEntry.headPtr;
}

/**
 * A beam that keeps track of several rules that may be further refined.
 */
class Beam final {
    private:

        uint32 numEntries_;

        BeamEntry* entries_;

        std::vector<std::reference_wrapper<BeamEntry>> order_;

    public:

        /**
         * @param beamWidth The maximum number of rules to keep track of
         */
        Beam(uint32 beamWidth) : numEntries_(beamWidth), entries_(new BeamEntry[numEntries_]) {
            order_.reserve(numEntries_);
        }

        /**
         * @param refinementComparator  A reference to an object of type `FixedRefinementComparator` that keeps track of
         *                              existing refinements of rules
         * @param thresholdsSubsetPtr   An unique pointer to an object of type `IThresholdsSubset` that has been used to
         *                              find the existing refinements of rules
         * @param labelIndices          A reference to an object of type `IIndexVector` that provides access to the
         *                              indices of the labels for which further refinement may predict
         * @param keepHeads             True, if further refinements should predict for the same labels as before, false
         *                              otherwise
         */
        Beam(FixedRefinementComparator& refinementComparator, std::unique_ptr<IThresholdsSubset> thresholdsSubsetPtr,
             const IIndexVector& labelIndices, bool keepHeads)
            : Beam(refinementComparator.getNumElements()) {
            FixedRefinementComparator::iterator iterator = refinementComparator.begin();
            uint32 i = 0;

            for (; i < numEntries_ - 1; i++) {
                Refinement& refinement = iterator[i];
                BeamEntry& entry = entries_[i];
                initializeEntry(entry, refinement, thresholdsSubsetPtr->copy(), labelIndices, keepHeads);
                order_.push_back(entry);
            }

            Refinement& refinement = iterator[i];
            BeamEntry& entry = entries_[i];
            initializeEntry(entry, refinement, std::move(thresholdsSubsetPtr), labelIndices, keepHeads);
            order_.push_back(entry);
        }

        ~Beam() {
            delete[] entries_;
        }

        /**
         * Searches for the best refinements of the rules that are kept track of by a given beam and updates the beam
         * accordingly.
         *
         * @param ruleCompareFunction   An object of type `RuleCompareFunction` that defines the function that should be
         *                              used for comparing the quality of different rules
         * @param beamPtr               A reference to an unique pointer of type `Beam` that represents the beam to be
         *                              updated
         * @param beamWidth             The number of rules the new beam should keep track of
         * @param featureSampling       A reference to an object of type `IFeatureSampling` that should be used for
         *                              sampling the features that may be used by potential refinements
         * @param keepHeads             True, if further refinements should predict for the same labels as before, false
         *                              otherwise
         * @param minCoverage           The number of training examples that must be covered by potential refinements
         * @param numThreads            The number of CPU threads to be used to search for potential refinements of a
         *                              rule in parallel
         * @param rng                   A reference to an object of type `RNG` that implements the random number
         *                              generator to be used
         * @return                      True, if any refinements have been found, false otherwise
         */
        static bool refine(RuleCompareFunction ruleCompareFunction, std::unique_ptr<Beam>& beamPtr, uint32 beamWidth,
                           IFeatureSampling& featureSampling, bool keepHeads, uint32 minCoverage, uint32 numThreads,
                           RNG& rng) {
            std::vector<std::reference_wrapper<BeamEntry>>& order = beamPtr->order_;
            std::unique_ptr<Beam> newBeamPtr = std::make_unique<Beam>(beamWidth);
            BeamEntry* newEntries = newBeamPtr->entries_;
            std::vector<std::reference_wrapper<BeamEntry>>& newOrder = newBeamPtr->order_;
            const BeamEntry& worstEntry = order.back();
            Quality minQuality(*worstEntry.headPtr);
            uint32 n = 0;
            bool result = false;

            // Traverse the existing beam entries....
            for (auto it = order.begin(); it != order.end(); it++) {
                BeamEntry& entry = *it;
                bool foundRefinement = false;

                // Check if existing beam entry can be refined...
                if (entry.labelIndices) {
                    // Sample features...
                    const IIndexVector& featureIndices = featureSampling.sample(rng);

                    // Search for refinements of the existing beam entry...
                    FixedRefinementComparator refinementComparator(ruleCompareFunction, beamWidth, minQuality);
                    foundRefinement = findRefinement(refinementComparator, *entry.thresholdsSubsetPtr, featureIndices,
                                                     *entry.labelIndices, minCoverage, numThreads);

                    if (foundRefinement) {
                        result = true;
                        uint32 numRefinements = refinementComparator.getNumElements();
                        FixedRefinementComparator::iterator iterator = refinementComparator.begin();
                        uint32 i = 0;

                        // Include all refinements, except for the last one, in the new beam. The corresponding
                        // `IThresholdsSubset` and `ConditionList` are copied...
                        for (; i < numRefinements - 1; i++) {
                            Refinement& refinement = iterator[i];

                            if (n < beamWidth) {
                                BeamEntry& newEntry = newEntries[n];
                                copyEntry(newEntry, entry, refinement, entry.thresholdsSubsetPtr->copy(),
                                          std::make_unique<ConditionList>(*entry.conditionListPtr), keepHeads,
                                          minCoverage);
                                newOrder.push_back(newEntry);
                                n++;
                            } else {
                                BeamEntry& newEntry = newOrder.back();
                                copyEntry(newEntry, entry, refinement, entry.thresholdsSubsetPtr->copy(),
                                          std::make_unique<ConditionList>(*entry.conditionListPtr), keepHeads,
                                          minCoverage);
                                minQuality = updateOrder(ruleCompareFunction, newOrder);
                            }
                        }

                        // Include the last refinement in the beam. The corresponding `IThresholdsSubset` and
                        // `ConditionList` are reused...
                        Refinement& refinement = iterator[i];

                        if (n < beamWidth) {
                            BeamEntry& newEntry = newEntries[n];
                            copyEntry(newEntry, entry, refinement, std::move(entry.thresholdsSubsetPtr),
                                      std::move(entry.conditionListPtr), keepHeads, minCoverage);
                            newOrder.push_back(newEntry);
                            n++;
                        } else {
                            BeamEntry& newEntry = newOrder.back();
                            copyEntry(newEntry, entry, refinement, std::move(entry.thresholdsSubsetPtr),
                                      std::move(entry.conditionListPtr), keepHeads, minCoverage);
                            minQuality = updateOrder(ruleCompareFunction, newOrder);
                        }
                    }
                }

                // If no refinement has been found, include the existing beam entry in the new beam unless it is worse
                // than the worst entry currently included. If there is a tie, the existing beam entry is preferred, as
                // it corresponds to a more general rule...
                if (!foundRefinement) {
                    if (n < beamWidth) {
                        BeamEntry& newEntry = newEntries[n];
                        copyEntry(newEntry, entry);
                        newOrder.push_back(newEntry);
                        n++;
                    } else if (!ruleCompareFunction.compare(minQuality, *entry.headPtr)) {
                        BeamEntry& newEntry = newOrder.back();
                        copyEntry(newEntry, entry);
                        minQuality = updateOrder(ruleCompareFunction, newOrder);
                    }
                }
            }

            newBeamPtr->numEntries_ = n;
            beamPtr = std::move(newBeamPtr);
            return result;
        }

        /**
         * Returns the entry that corresponds to the best rule that is currently kept track of by the beam.
         *
         * @return A reference to an object of type `BeamEntry` that corresponds to the best rule
         */
        BeamEntry& getBestEntry() {
            return order_.front();
        }
};

/**
 * An implementation of the type `IRuleInduction` that allows to induce individual rules by using a top-down beam
 * search.
 */
class BeamSearchTopDownRuleInduction final : public AbstractRuleInduction {
    private:

        const RuleCompareFunction ruleCompareFunction_;

        const uint32 beamWidth_;

        const bool resampleFeatures_;

        const uint32 minCoverage_;

        const uint32 maxConditions_;

        const uint32 maxHeadRefinements_;

        const uint32 numThreads_;

    public:

        /**
         * @param ruleCompareFunction       An object of type `RuleCompareFunction` that defines the function that
         *                                  should be used for comparing the quality of different rules
         * @param beamWidth                 The width that should be used by the beam search. Must be at least 2
         * @param resampleFeatures          True, if a new sample of the available features should be created for each
         *                                  rule that is refined during the beam search, false otherwise
         * @param minCoverage               The minimum number of training examples that must be covered by a rule. Must
         *                                  be at least 1
         * @param maxConditions             The maximum number of conditions to be included in a rule's body. Must be at
         *                                  least 2 or 0, if the number of conditions should not be restricted
         * @param maxHeadRefinements        The maximum number of times, the head of a rule may be refinement after a
         *                                  new condition has been added to its body. Must be at least 1 or 0, if the
         *                                  number of refinements should not be restricted
         * @param recalculatePredictions    True, if the predictions of rules should be recalculated on all training
         *                                  examples, if some of the examples have zero weights, false otherwise
         * @param numThreads                The number of CPU threads to be used to search for potential refinements of
         *                                  a rule in parallel. Must be at least 1
         */
        BeamSearchTopDownRuleInduction(RuleCompareFunction ruleCompareFunction, uint32 beamWidth, bool resampleFeatures,
                                       uint32 minCoverage, uint32 maxConditions, uint32 maxHeadRefinements,
                                       bool recalculatePredictions, uint32 numThreads)
            : AbstractRuleInduction(recalculatePredictions), ruleCompareFunction_(ruleCompareFunction),
              beamWidth_(beamWidth), resampleFeatures_(resampleFeatures), minCoverage_(minCoverage),
              maxConditions_(maxConditions), maxHeadRefinements_(maxHeadRefinements), numThreads_(numThreads) {}

    protected:

        std::unique_ptr<IThresholdsSubset> growRule(
          IThresholds& thresholds, const IIndexVector& labelIndices, const IWeightVector& weights,
          IPartition& partition, IFeatureSampling& featureSampling, RNG& rng,
          std::unique_ptr<ConditionList>& conditionListPtr,
          std::unique_ptr<AbstractEvaluatedPrediction>& headPtr) const override {
            // Create a new subset of the given thresholds...
            std::unique_ptr<IThresholdsSubset> thresholdsSubsetPtr = weights.createThresholdsSubset(thresholds);

            // Sample features...
            const IIndexVector& sampledFeatureIndices = featureSampling.sample(rng);

            // Search for the best refinements using a single condition...
            FixedRefinementComparator refinementComparator(ruleCompareFunction_, beamWidth_);
            bool foundRefinement = findRefinement(refinementComparator, *thresholdsSubsetPtr, sampledFeatureIndices,
                                                  labelIndices, minCoverage_, numThreads_);

            if (foundRefinement) {
                bool keepHeads = maxHeadRefinements_ == 1;
                std::unique_ptr<Beam> beamPtr =
                  std::make_unique<Beam>(refinementComparator, std::move(thresholdsSubsetPtr), labelIndices, keepHeads);
                uint32 searchDepth = 1;

                while (foundRefinement && (maxConditions_ == 0 || searchDepth < maxConditions_)) {
                    searchDepth++;
                    keepHeads = maxHeadRefinements_ > 0 && searchDepth >= maxHeadRefinements_;

                    // Create a `IFeatureSampling` to be used for refining the current beam...
                    std::unique_ptr<IFeatureSampling> beamSearchFeatureSamplingPtr =
                      featureSampling.createBeamSearchFeatureSampling(rng, resampleFeatures_);

                    // Search for the best refinements within the current beam...
                    foundRefinement =
                      beamPtr->refine(ruleCompareFunction_, beamPtr, beamWidth_, *beamSearchFeatureSamplingPtr,
                                      keepHeads, minCoverage_, numThreads_, rng);
                }

                BeamEntry& entry = beamPtr->getBestEntry();
                conditionListPtr = std::move(entry.conditionListPtr);
                headPtr = std::move(entry.headPtr);
                return std::move(entry.thresholdsSubsetPtr);
            }

            return thresholdsSubsetPtr;
        }
};

/**
 * Allows to create instances of the type `IRuleInduction` that induce individual rules by using a top-down beam search,
 * where new conditions are added iteratively to the (initially empty) body of a rule. At each iteration, the refinement
 * that improves the rule the most is chosen. The search stops if no refinement results in an improvement.
 */
class BeamSearchTopDownRuleInductionFactory final : public IRuleInductionFactory {
    private:

        const RuleCompareFunction ruleCompareFunction_;

        const uint32 beamWidth_;

        const bool resampleFeatures_;

        const uint32 minCoverage_;

        const uint32 maxConditions_;

        const uint32 maxHeadRefinements_;

        const bool recalculatePredictions_;

        const uint32 numThreads_;

    public:

        /**
         * @param ruleCompareFunction       An object of type `RuleCompareFunction` that defines the function that
         *                                  should be used for comparing the quality of different rules
         * @param beamWidth                 The width that should be used by the beam search. Must be at least 2
         * @param resampleFeatures          True, if a new sample of the available features should be created for each
         *                                  rule that is refined during the beam search, false otherwise
         * @param minCoverage               The minimum number of training examples that must be covered by a rule. Must
         *                                  be at least 1
         * @param maxConditions             The maximum number of conditions to be included in a rule's body. Must be at
         *                                  least 2 or 0, if the number of conditions should not be restricted
         * @param maxHeadRefinements        The maximum number of times, the head of a rule may be refined after a new
         *                                  condition has been added to its body. Must be at least 1 or 0, if the number
         *                                  of refinements should not be restricted
         * @param recalculatePredictions    True, if the predictions of rules should be recalculated on all training
         *                                  examples, if some of the examples have zero weights, false otherwise
         * @param numThreads                The number of CPU threads to be used to search for potential refinements of
         *                                  a rule in parallel. Must be at least 1
         */
        BeamSearchTopDownRuleInductionFactory(RuleCompareFunction ruleCompareFunction, uint32 beamWidth,
                                              bool resampleFeatures, uint32 minCoverage, uint32 maxConditions,
                                              uint32 maxHeadRefinements, bool recalculatePredictions, uint32 numThreads)
            : ruleCompareFunction_(ruleCompareFunction), beamWidth_(beamWidth), resampleFeatures_(resampleFeatures),
              minCoverage_(minCoverage), maxConditions_(maxConditions), maxHeadRefinements_(maxHeadRefinements),
              recalculatePredictions_(recalculatePredictions), numThreads_(numThreads) {}

        std::unique_ptr<IRuleInduction> create() const override {
            return std::make_unique<BeamSearchTopDownRuleInduction>(ruleCompareFunction_, beamWidth_, resampleFeatures_,
                                                                    minCoverage_, maxConditions_, maxHeadRefinements_,
                                                                    recalculatePredictions_, numThreads_);
        }
};

BeamSearchTopDownRuleInductionConfig::BeamSearchTopDownRuleInductionConfig(
  RuleCompareFunction ruleCompareFunction, const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
    : ruleCompareFunction_(ruleCompareFunction), beamWidth_(4), resampleFeatures_(false), minCoverage_(1),
      minSupport_(0.0f), maxConditions_(0), maxHeadRefinements_(1), recalculatePredictions_(true),
      multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

uint32 BeamSearchTopDownRuleInductionConfig::getBeamWidth() const {
    return beamWidth_;
}

IBeamSearchTopDownRuleInductionConfig& BeamSearchTopDownRuleInductionConfig::setBeamWidth(uint32 beamWidth) {
    assertGreaterOrEqual<uint32>("beamWidth", beamWidth, 2);
    beamWidth_ = beamWidth;
    return *this;
}

bool BeamSearchTopDownRuleInductionConfig::areFeaturesResampled() const {
    return resampleFeatures_;
}

IBeamSearchTopDownRuleInductionConfig& BeamSearchTopDownRuleInductionConfig::setResampleFeatures(
  bool resampleFeatures) {
    resampleFeatures_ = resampleFeatures;
    return *this;
}

uint32 BeamSearchTopDownRuleInductionConfig::getMinCoverage() const {
    return minCoverage_;
}

IBeamSearchTopDownRuleInductionConfig& BeamSearchTopDownRuleInductionConfig::setMinCoverage(uint32 minCoverage) {
    assertGreaterOrEqual<uint32>("minCoverage", minCoverage, 1);
    minCoverage_ = minCoverage;
    return *this;
}

float32 BeamSearchTopDownRuleInductionConfig::getMinSupport() const {
    return minSupport_;
}

IBeamSearchTopDownRuleInductionConfig& BeamSearchTopDownRuleInductionConfig::setMinSupport(float32 minSupport) {
    if (minSupport != 0) assertGreater<float32>("minSupport", minSupport, 0);
    if (minSupport != 0) assertLess<float32>("minSupport", minSupport, 1);
    minSupport_ = minSupport;
    return *this;
}

uint32 BeamSearchTopDownRuleInductionConfig::getMaxConditions() const {
    return maxConditions_;
}

IBeamSearchTopDownRuleInductionConfig& BeamSearchTopDownRuleInductionConfig::setMaxConditions(uint32 maxConditions) {
    if (maxConditions != 0) assertGreaterOrEqual<uint32>("maxConditions", maxConditions, 2);
    maxConditions_ = maxConditions;
    return *this;
}

uint32 BeamSearchTopDownRuleInductionConfig::getMaxHeadRefinements() const {
    return maxHeadRefinements_;
}

IBeamSearchTopDownRuleInductionConfig& BeamSearchTopDownRuleInductionConfig::setMaxHeadRefinements(
  uint32 maxHeadRefinements) {
    if (maxHeadRefinements != 0) assertGreaterOrEqual<uint32>("maxHeadRefinements", maxHeadRefinements, 1);
    maxHeadRefinements_ = maxHeadRefinements;
    return *this;
}

bool BeamSearchTopDownRuleInductionConfig::arePredictionsRecalculated() const {
    return recalculatePredictions_;
}

IBeamSearchTopDownRuleInductionConfig& BeamSearchTopDownRuleInductionConfig::setRecalculatePredictions(
  bool recalculatePredictions) {
    recalculatePredictions_ = recalculatePredictions;
    return *this;
}

std::unique_ptr<IRuleInductionFactory> BeamSearchTopDownRuleInductionConfig::createRuleInductionFactory(
  const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix) const {
    uint32 numExamples = featureMatrix.getNumRows();
    uint32 minCoverage;

    if (minSupport_ > 0) {
        minCoverage = calculateBoundedFraction(numExamples, minSupport_, minCoverage_, numExamples);
    } else {
        minCoverage = std::min(numExamples, minCoverage_);
    }

    uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
    return std::make_unique<BeamSearchTopDownRuleInductionFactory>(ruleCompareFunction_, beamWidth_, resampleFeatures_,
                                                                   minCoverage, maxConditions_, maxHeadRefinements_,
                                                                   recalculatePredictions_, numThreads);
}
