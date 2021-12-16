#include "common/thresholds/thresholds_approximate.hpp"
#include "common/binning/feature_binning_nominal.hpp"
#include "common/rule_refinement/rule_refinement_approximate.hpp"
#include "common/validation.hpp"
#include "thresholds_common.hpp"
#include <unordered_map>

/**
 * An entry that is stored in the cache. It contains the result of a binning method and an unique pointer to an
 * histogram, well as to a vector that stores the weights of individual bins.
 */
struct CacheEntry : public IFeatureBinning::Result {

    /**
     * An unique pointer to an object of type `IHistogram` that provides access to the values stored in a histogram.
     */
    std::unique_ptr<IHistogram> histogramPtr;

    /**
     * An unique pointer to an object of type `BinWeightVector` that provides access to the weights of individual bins.
     */
    std::unique_ptr<BinWeightVector> weightVectorPtr;

};

/**
 * Updates a given `CoverageSet` after a new condition has been added, such that only the examples that are covered by
 * the new rule are marked es covered.
 *
 * @param thresholdVector   A reference to an object of type `ThresholdVector` that stores the thresholds that result
 *                          from the boundaries of the bins
 * @param binIndices        A reference to an object of type `IBinIndexVector` that stores the indices of the bins,
 *                          individual examples belong to
 * @param conditionStart    The first bin (inclusive) that is covered by the new condition
 * @param conditionEnd      The last bin (exclusive) that is covered by the new condition
 * @param covered           True, if the bins in range [conditionStart, conditionEnd) are covered by the new condition
 *                          and the remaining ones are not, false, if the elements in said range are not covered, but
 *                          the remaining ones are
 * @param coverageSet       A reference to an object of type `CoverageSet` that is used to keep track of the examples
 *                          that are covered by the previous rule. It will be updated by this function
 * @param statistics        A reference to an object of type `IStatistics` to be notified about the statistics that must
 *                          be considered when searching for the next refinement, i.e., the statistics that are covered
 *                          by the new rule
 * @param weights           A reference to an an object of type `IWeightVector` that provides access to the weights of
 *                          the individual training examples
 */
static inline void updateCoveredExamples(const ThresholdVector& thresholdVector, const IBinIndexVector& binIndices,
                                         int64 conditionStart, int64 conditionEnd, bool covered,
                                         CoverageSet& coverageSet, IStatistics& statistics,
                                         const IWeightVector& weights) {
    int64 start, end;

    if (conditionEnd < conditionStart) {
        start = conditionEnd + 1;
        end = conditionStart + 1;
    } else {
        start = conditionStart;
        end = conditionEnd;
    }

    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::iterator coverageSetIterator = coverageSet.begin();
    statistics.resetCoveredStatistics();
    uint32 n = 0;

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = coverageSetIterator[i];

        if (!thresholdVector.isMissing(exampleIndex)) {
            uint32 binIndex = binIndices.getBinIndex(exampleIndex);

            if (binIndex == IBinIndexVector::BIN_INDEX_SPARSE) {
                binIndex = thresholdVector.getSparseBinIndex();
            }

            if ((binIndex >= start && binIndex < end) == covered) {
                float64 weight = weights.getWeight(exampleIndex);
                statistics.updateCoveredStatistic(exampleIndex, weight, false);
                coverageSetIterator[n] = exampleIndex;
                n++;
            }
        }
    }

    coverageSet.setNumCovered(n);
}

/**
 * Rebuilds a given histogram such that is contains the statistics of all examples that are currently covered and
 * updates the weights of the individual bins.
 *
 * @param thresholdVector   A reference to an object of type `ThresholdVector` that stores the thresholds that result
 *                          from the boundaries of the bins
 * @param binIndices        A reference to an object of type `IBinIndexVector` that stores the indices of the bins,
 *                          individual examples belong to
 * @param binWeights        A reference to an object of type `BinWeightVector` that stores the weights of individual
 *                          bins
 * @param histogram         A reference to an object of type `IHistogram` that should be rebuild
 * @param weights           A reference to an an object of type `IWeightVector` that provides access to the weights of
 *                          the individual training examples
 * @param coverageSet       A reference to an object of type `CoverageSet` that is used to keep track of the examples
 *                          that are currently covered
 */
static inline void rebuildHistogram(const ThresholdVector& thresholdVector, const IBinIndexVector& binIndices,
                                    BinWeightVector& binWeights, IHistogram& histogram, const IWeightVector& weights,
                                    const CoverageSet& coverageSet) {
    // Reset all statistics in the histogram to zero...
    histogram.clear();

    // Reset the weights of all bins to zero...
    binWeights.clear();

    // Iterate the covered examples and add their statistics to the corresponding bin...
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator coverageSetIterator = coverageSet.cbegin();
    bool sparseBinWeight = false;

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = coverageSetIterator[i];

        if (!thresholdVector.isMissing(exampleIndex)) {
            float64 weight = weights.getWeight(exampleIndex);

            if (weight > 0) {
                uint32 binIndex = binIndices.getBinIndex(exampleIndex);

                if (binIndex != IBinIndexVector::BIN_INDEX_SPARSE) {
                    binWeights.set(binIndex, true);
                    histogram.addToBin(binIndex, exampleIndex, weight);
                } else {
                    sparseBinWeight = true;
                }
            }
        }
    }

    uint32 sparseBinIndex = thresholdVector.getSparseBinIndex();

    if (sparseBinIndex < thresholdVector.getNumElements()) {
        binWeights.set(sparseBinIndex, sparseBinWeight);
    }
}

/**
 * Provides access to the thresholds that result from applying a binning method to the feature values of the training
 * examples.
 */
class ApproximateThresholds final : public AbstractThresholds {

    private:

        /**
         * Provides access to a subset of the thresholds that are stored by an instance of the class
         * `ApproximateThresholds`.
         */
        class ThresholdsSubset final : public IThresholdsSubset {

            private:

                /**
                 * A callback that allows to retrieve bins and corresponding statistics. If available, the bins and
                 * statistics are retrieved from the cache. Otherwise, they are computed by fetching the feature values
                 * from the feature matrix and applying a binning method.
                 */
                class Callback final : public IRuleRefinementCallback<ThresholdVector, BinWeightVector> {

                    private:

                        ThresholdsSubset& thresholdsSubset_;

                        uint32 featureIndex_;

                        bool nominal_;

                    public:

                        /**
                         * @param thresholdsSubset  A reference to an object of type `ThresholdsSubset` that caches the
                         *                          bins
                         * @param featureIndex      The index of the feature for which the bins should be retrieved
                         * @param nominal           True, if the feature at index `featureIndex` is nominal, false
                         *                          otherwise
                         */
                        Callback(ThresholdsSubset& thresholdsSubset, uint32 featureIndex, bool nominal)
                            : thresholdsSubset_(thresholdsSubset), featureIndex_(featureIndex), nominal_(nominal) {

                        }

                        std::unique_ptr<Result> get() override {
                            auto cacheIterator = thresholdsSubset_.thresholds_.cache_.find(featureIndex_);
                            ThresholdVector* thresholdVector = cacheIterator->second.thresholdVectorPtr.get();
                            IBinIndexVector* binIndices = cacheIterator->second.binIndicesPtr.get();

                            if (thresholdVector == nullptr) {
                                // Fetch feature vector...
                                std::unique_ptr<FeatureVector> featureVectorPtr;
                                const IFeatureMatrix& featureMatrix = thresholdsSubset_.thresholds_.featureMatrix_;
                                uint32 numExamples = featureMatrix.getNumRows();
                                featureMatrix.fetchFeatureVector(featureIndex_, featureVectorPtr);

                                // Apply binning method...
                                const IFeatureBinning& binning =
                                    nominal_ ? thresholdsSubset_.thresholds_.nominalBinning_
                                             : thresholdsSubset_.thresholds_.binning_;
                                IFeatureBinning::Result result = binning.createBins(*featureVectorPtr, numExamples);
                                cacheIterator->second.thresholdVectorPtr = std::move(result.thresholdVectorPtr);
                                thresholdVector = cacheIterator->second.thresholdVectorPtr.get();
                                cacheIterator->second.binIndicesPtr = std::move(result.binIndicesPtr);
                                binIndices = cacheIterator->second.binIndicesPtr.get();

                                // Create histogram and weight vector...
                                uint32 numBins = thresholdVector->getNumElements();
                                cacheIterator->second.histogramPtr =
                                    thresholdsSubset_.thresholds_.statisticsProvider_.get().createHistogram(numBins);
                                cacheIterator->second.weightVectorPtr = std::make_unique<BinWeightVector>(numBins);
                            }

                            // Rebuild histogram...
                            IHistogram& histogram = *cacheIterator->second.histogramPtr;
                            BinWeightVector& binWeights = *cacheIterator->second.weightVectorPtr;
                            rebuildHistogram(*thresholdVector, *binIndices, binWeights, histogram,
                                             thresholdsSubset_.weights_, thresholdsSubset_.coverageSet_);

                            return std::make_unique<Result>(histogram, binWeights, *thresholdVector);
                        }

                };

                ApproximateThresholds& thresholds_;

                const IWeightVector& weights_;

                CoverageSet coverageSet_;

                template<typename T>
                std::unique_ptr<IRuleRefinement> createApproximateRuleRefinement(const T& labelIndices,
                                                                                 uint32 featureIndex) {
                    thresholds_.cache_.emplace(featureIndex, CacheEntry());
                    bool nominal = thresholds_.nominalFeatureMask_.isNominal(featureIndex);
                    std::unique_ptr<Callback> callbackPtr = std::make_unique<Callback>(*this, featureIndex, nominal);
                    return std::make_unique<ApproximateRuleRefinement<T>>(labelIndices, featureIndex, nominal, weights_,
                                                                          std::move(callbackPtr));
                }

            public:

                /**
                 * @param thresholds    A reference to an object of type `ApproximateThresholds` that stores the
                 *                      thresholds
                 * @param weights       A reference to an object of type `IWeightWeight` that provides access to the
                 *                      weights of individual training examples
                 */
                ThresholdsSubset(ApproximateThresholds& thresholds, const IWeightVector& weights)
                    : thresholds_(thresholds), weights_(weights),
                      coverageSet_(CoverageSet(thresholds.getNumExamples())) {

                }

                std::unique_ptr<IRuleRefinement> createRuleRefinement(const CompleteIndexVector& labelIndices,
                                                                      uint32 featureIndex) override {
                    return createApproximateRuleRefinement(labelIndices, featureIndex);
                }

                std::unique_ptr<IRuleRefinement> createRuleRefinement(const PartialIndexVector& labelIndices,
                                                                      uint32 featureIndex) override {
                    return createApproximateRuleRefinement(labelIndices, featureIndex);
                }

                void filterThresholds(Refinement& refinement) override {
                    uint32 featureIndex = refinement.featureIndex;
                    auto cacheIterator = thresholds_.cache_.find(featureIndex);
                    const ThresholdVector& thresholdVector = *cacheIterator->second.thresholdVectorPtr;
                    const IBinIndexVector& binIndices = *cacheIterator->second.binIndicesPtr;
                    updateCoveredExamples(thresholdVector, binIndices, refinement.start, refinement.end,
                                          refinement.covered, coverageSet_, thresholds_.statisticsProvider_.get(),
                                          weights_);
                }

                void filterThresholds(const Condition& condition) override {
                    uint32 featureIndex = condition.featureIndex;
                    auto cacheIterator = thresholds_.cache_.find(featureIndex);
                    const ThresholdVector& thresholdVector = *cacheIterator->second.thresholdVectorPtr;
                    const IBinIndexVector& binIndices = *cacheIterator->second.binIndicesPtr;
                    updateCoveredExamples(thresholdVector, binIndices, condition.start, condition.end,
                                          condition.covered, coverageSet_, thresholds_.statisticsProvider_.get(),
                                          weights_);
                }

                void resetThresholds() override {
                    coverageSet_.reset();
                }

                const ICoverageState& getCoverageState() const override {
                    return coverageSet_;
                }

                float64 evaluateOutOfSample(const SinglePartition& partition, const CoverageMask& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally<SinglePartition::const_iterator>(
                        partition.cbegin(), partition.getNumElements(), weights_, coverageState,
                        thresholds_.statisticsProvider_.get(), head);
                }

                float64 evaluateOutOfSample(const BiPartition& partition, const CoverageMask& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally<BiPartition::const_iterator>(
                        partition.first_cbegin(), partition.getNumFirst(), weights_, coverageState,
                        thresholds_.statisticsProvider_.get(), head);
                }

                float64 evaluateOutOfSample(const SinglePartition& partition, const CoverageSet& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally(weights_, coverageState, thresholds_.statisticsProvider_.get(),
                                                         head);
                }

                float64 evaluateOutOfSample(BiPartition& partition, const CoverageSet& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally(weights_, coverageState, partition,
                                                         thresholds_.statisticsProvider_.get(), head);
                }

                void recalculatePrediction(const SinglePartition& partition, const CoverageMask& coverageState,
                                           Refinement& refinement) const override {
                    recalculatePredictionInternally<SinglePartition::const_iterator>(
                        partition.cbegin(), partition.getNumElements(), coverageState,
                        thresholds_.statisticsProvider_.get(), refinement);
                }

                void recalculatePrediction(const BiPartition& partition, const CoverageMask& coverageState,
                                           Refinement& refinement) const override {
                    recalculatePredictionInternally<BiPartition::const_iterator>(
                        partition.first_cbegin(), partition.getNumFirst(), coverageState,
                        thresholds_.statisticsProvider_.get(), refinement);
                }

                void recalculatePrediction(const SinglePartition& partition, const CoverageSet& coverageState,
                                           Refinement& refinement) const override {
                    recalculatePredictionInternally(coverageState, thresholds_.statisticsProvider_.get(), refinement);
                }

                void recalculatePrediction(BiPartition& partition, const CoverageSet& coverageState,
                                           Refinement& refinement) const override {
                    recalculatePredictionInternally(coverageState, partition, thresholds_.statisticsProvider_.get(),
                                                    refinement);
                }

                void applyPrediction(const AbstractPrediction& prediction) override {
                    uint32 numCovered = coverageSet_.getNumCovered();
                    CoverageSet::const_iterator iterator = coverageSet_.cbegin();
                    const AbstractPrediction* predictionPtr = &prediction;
                    IStatistics* statisticsPtr = &thresholds_.statisticsProvider_.get();
                    uint32 numThreads = thresholds_.numThreads_;

                    #pragma omp parallel for firstprivate(numCovered) firstprivate(iterator) \
                    firstprivate(predictionPtr) firstprivate(statisticsPtr) schedule(dynamic) num_threads(numThreads)
                    for (int64 i = 0; i < numCovered; i++) {
                        uint32 exampleIndex = iterator[i];
                        predictionPtr->apply(*statisticsPtr, exampleIndex);
                    }
                }

        };

        NominalFeatureBinning nominalBinning_;

        const IFeatureBinning& binning_;

        uint32 numThreads_;

        std::unordered_map<uint32, CacheEntry> cache_;

    public:

        /**
         * @param featureMatrix         A reference to an object of type `IFeatureMatrix` that provides access to the
         *                              feature values of the training examples
         * @param nominalFeatureMask    A reference  to an object of type `INominalFeatureMask` that provides access to
         *                              the information whether individual features are nominal or not
         * @param statisticsProvider    A reference to an object of type `IStatisticsProvider` that provides access to
         *                              statistics about the labels of the training examples
         * @param binning               A reference to an object of type `IFeatureBinning` that implements the binning
         *                              method to be used
         * @param numThreads            The number of CPU threads to be used to update statistics in parallel
         */
        ApproximateThresholds(const IFeatureMatrix& featureMatrix, const INominalFeatureMask& nominalFeatureMask,
                              IStatisticsProvider& statisticsProvider, const IFeatureBinning& binning,
                              uint32 numThreads)
            : AbstractThresholds(featureMatrix, nominalFeatureMask, statisticsProvider),
              binning_(binning), numThreads_(numThreads) {

        }

        std::unique_ptr<IThresholdsSubset> createSubset(const IWeightVector& weights) override {
            updateSampledStatisticsInternally(statisticsProvider_.get(), weights);
            return std::make_unique<ApproximateThresholds::ThresholdsSubset>(*this, weights);
        }

};

ApproximateThresholdsFactory::ApproximateThresholdsFactory(std::unique_ptr<IFeatureBinning> binningPtr,
                                                           uint32 numThreads)
    : binningPtr_(std::move(binningPtr)), numThreads_(numThreads) {
    assertNotNull("binningPtr", binningPtr_.get());
    assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
}

std::unique_ptr<IThresholds> ApproximateThresholdsFactory::create(
        const IFeatureMatrix& featureMatrix, const INominalFeatureMask& nominalFeatureMask,
        IStatisticsProvider& statisticsProvider) const {
    return std::make_unique<ApproximateThresholds>(featureMatrix, nominalFeatureMask, statisticsProvider, *binningPtr_,
                                                   numThreads_);
}
