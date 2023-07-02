#include "common/thresholds/thresholds_approximate.hpp"

#include "common/rule_refinement/rule_refinement_approximate.hpp"
#include "thresholds_common.hpp"

#include <unordered_map>

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
 * @param statistics        A reference to an object of type `IWeightedStatistics` to be notified about the statistics
 *                          that must be considered when searching for the next refinement, i.e., the statistics that
 *                          are covered by the new rule
 */
static inline void updateCoveredExamples(const ThresholdVector& thresholdVector, const IBinIndexVector& binIndices,
                                         int64 conditionStart, int64 conditionEnd, bool covered,
                                         CoverageSet& coverageSet, IWeightedStatistics& statistics) {
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
                statistics.addCoveredStatistic(exampleIndex);
                coverageSetIterator[n] = exampleIndex;
                n++;
            }
        }
    }

    coverageSet.setNumCovered(n);
}

/**
 * Rebuilds a given histogram.
 *
 * @param thresholdVector   A reference to an object of type `ThresholdVector` that stores the thresholds that result
 *                          from the boundaries of the bins
 * @param histogram         A reference to an object of type `IHistogram` that should be rebuild
 * @param coverageSet       A reference to an object of type `CoverageSet` that is used to keep track of the examples
 *                          that are currently covered
 */
static inline void rebuildHistogram(const ThresholdVector& thresholdVector, IHistogram& histogram,
                                    const CoverageSet& coverageSet) {
    // Reset all statistics in the histogram to zero...
    histogram.clear();

    // Iterate the covered examples and add their statistics to the corresponding bin...
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator coverageSetIterator = coverageSet.cbegin();

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = coverageSetIterator[i];

        if (!thresholdVector.isMissing(exampleIndex)) {
            histogram.addToBin(exampleIndex);
        }
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
         *
         * @tparam WeightVector The type of the vector that provides access to the weights of individual training
         *                      examples
         */
        template<typename WeightVector>
        class ThresholdsSubset final : public IThresholdsSubset {
            private:

                /**
                 * A callback that allows to retrieve bins and corresponding statistics. If available, the bins and
                 * statistics are retrieved from the cache. Otherwise, they are computed by fetching the feature values
                 * from the feature matrix and applying a binning method.
                 */
                class Callback final : public IRuleRefinementCallback<IHistogram, ThresholdVector> {
                    private:

                        ThresholdsSubset& thresholdsSubset_;

                        const uint32 featureIndex_;

                        const bool nominal_;

                    public:

                        /**
                         * @param thresholdsSubset  A reference to an object of type `ThresholdsSubset` that caches the
                         *                          bins
                         * @param featureIndex      The index of the feature for which the bins should be retrieved
                         * @param nominal           True, if the feature at index `featureIndex` is nominal, false
                         *                          otherwise
                         */
                        Callback(ThresholdsSubset& thresholdsSubset, uint32 featureIndex, bool nominal)
                            : thresholdsSubset_(thresholdsSubset), featureIndex_(featureIndex), nominal_(nominal) {}

                        Result get() override {
                            auto cacheIterator = thresholdsSubset_.thresholds_.cache_.find(featureIndex_);
                            IFeatureBinning::Result& cacheEntry = cacheIterator->second;
                            ThresholdVector* thresholdVector = cacheEntry.thresholdVectorPtr.get();
                            IBinIndexVector* binIndices = cacheEntry.binIndicesPtr.get();

                            if (!thresholdVector) {
                                // Fetch feature vector...
                                std::unique_ptr<FeatureVector> featureVectorPtr;
                                const IColumnWiseFeatureMatrix& featureMatrix =
                                  thresholdsSubset_.thresholds_.featureMatrix_;
                                uint32 numExamples = featureMatrix.getNumRows();
                                featureMatrix.fetchFeatureVector(featureIndex_, featureVectorPtr);

                                // Apply binning method...
                                const IFeatureBinning& binning =
                                  nominal_ ? *thresholdsSubset_.thresholds_.nominalFeatureBinningPtr_
                                           : *thresholdsSubset_.thresholds_.numericalFeatureBinningPtr_;
                                IFeatureBinning::Result result = binning.createBins(*featureVectorPtr, numExamples);
                                cacheEntry.thresholdVectorPtr = std::move(result.thresholdVectorPtr);
                                thresholdVector = cacheEntry.thresholdVectorPtr.get();
                                cacheEntry.binIndicesPtr = std::move(result.binIndicesPtr);
                                binIndices = cacheEntry.binIndicesPtr.get();
                            }

                            auto cacheHistogramIterator = thresholdsSubset_.cacheHistogram_.find(featureIndex_);

                            if (!cacheHistogramIterator->second) {
                                // Create histogram and weight vector...
                                uint32 numBins = thresholdVector->getNumElements();
                                cacheHistogramIterator->second =
                                  binIndices->createHistogram(*thresholdsSubset_.weightedStatisticsPtr_, numBins);
                            }

                            // Rebuild histogram...
                            IHistogram& histogram = *cacheHistogramIterator->second;
                            rebuildHistogram(*thresholdVector, histogram, thresholdsSubset_.coverageSet_);

                            return Result(histogram, *thresholdVector);
                        }
                };

                ApproximateThresholds& thresholds_;

                std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr_;

                const WeightVector& weights_;

                CoverageSet coverageSet_;

                std::unordered_map<uint32, std::unique_ptr<IHistogram>> cacheHistogram_;

                template<typename IndexVector>
                std::unique_ptr<IRuleRefinement> createApproximateRuleRefinement(const IndexVector& labelIndices,
                                                                                 uint32 featureIndex) {
                    // Retrieve `unique_ptr` from the cache, or insert an empty one if it does not already exist...
                    auto cacheHistogramIterator =
                      cacheHistogram_.emplace(featureIndex, std::unique_ptr<IHistogram>()).first;

                    // If the `unique_ptr` in the cache does not refer to an `IHistogram`, add an empty
                    // `IFeatureBinning::Result` to the cache...
                    if (!cacheHistogramIterator->second) {
                        thresholds_.cache_.emplace(featureIndex, IFeatureBinning::Result());
                    }

                    std::unique_ptr<IFeatureType> featureTypePtr =
                      thresholds_.featureInfo_.createFeatureType(featureIndex);
                    bool nominal = featureTypePtr->isNominal();
                    std::unique_ptr<Callback> callbackPtr = std::make_unique<Callback>(*this, featureIndex, nominal);
                    return std::make_unique<ApproximateRuleRefinement<IndexVector>>(
                      labelIndices, coverageSet_.getNumCovered(), featureIndex, nominal, std::move(callbackPtr));
                }

            public:

                /**
                 * @param thresholds            A reference to an object of type `ApproximateThresholds` that stores the
                 *                              thresholds
                 * @param weightedStatisticsPtr An unique pointer to an object of type `IWeightedStatistics` that
                 *                              provides access to the statistics
                 * @param weights               A reference to an object of template type `WeightWeight` that provides
                 *                              access to the weights of individual training examples
                 */
                ThresholdsSubset(ApproximateThresholds& thresholds,
                                 std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr,
                                 const WeightVector& weights)
                    : thresholds_(thresholds), weightedStatisticsPtr_(std::move(weightedStatisticsPtr)),
                      weights_(weights), coverageSet_(CoverageSet(thresholds.featureMatrix_.getNumRows())) {}

                /**
                 * @param thresholdsSubset A reference to an object of type `ThresholdsSubset` to be copied
                 */
                ThresholdsSubset(const ThresholdsSubset& thresholdsSubset)
                    : thresholds_(thresholdsSubset.thresholds_),
                      weightedStatisticsPtr_(thresholdsSubset.weightedStatisticsPtr_->copy()),
                      weights_(thresholdsSubset.weights_), coverageSet_(CoverageSet(thresholdsSubset.coverageSet_)) {}

                std::unique_ptr<IThresholdsSubset> copy() const override {
                    return std::make_unique<ThresholdsSubset<WeightVector>>(*this);
                }

                std::unique_ptr<IRuleRefinement> createRuleRefinement(const CompleteIndexVector& labelIndices,
                                                                      uint32 featureIndex) override {
                    return createApproximateRuleRefinement(labelIndices, featureIndex);
                }

                std::unique_ptr<IRuleRefinement> createRuleRefinement(const PartialIndexVector& labelIndices,
                                                                      uint32 featureIndex) override {
                    return createApproximateRuleRefinement(labelIndices, featureIndex);
                }

                void filterThresholds(const Condition& condition) override {
                    uint32 featureIndex = condition.featureIndex;
                    auto cacheIterator = thresholds_.cache_.find(featureIndex);
                    IFeatureBinning::Result& cacheEntry = cacheIterator->second;
                    const ThresholdVector& thresholdVector = *cacheEntry.thresholdVectorPtr;
                    const IBinIndexVector& binIndices = *cacheEntry.binIndicesPtr;
                    updateCoveredExamples(thresholdVector, binIndices, condition.start, condition.end,
                                          condition.covered, coverageSet_, *weightedStatisticsPtr_);
                }

                void resetThresholds() override {
                    coverageSet_.reset();
                }

                const ICoverageState& getCoverageState() const override {
                    return coverageSet_;
                }

                Quality evaluateOutOfSample(const SinglePartition& partition, const CoverageMask& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally<SinglePartition::const_iterator>(
                      partition.cbegin(), partition.getNumElements(), weights_, coverageState,
                      thresholds_.statisticsProvider_.get(), head);
                }

                Quality evaluateOutOfSample(const BiPartition& partition, const CoverageMask& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally<BiPartition::const_iterator>(
                      partition.first_cbegin(), partition.getNumFirst(), weights_, coverageState,
                      thresholds_.statisticsProvider_.get(), head);
                }

                Quality evaluateOutOfSample(const SinglePartition& partition, const CoverageSet& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally(weights_, coverageState, thresholds_.statisticsProvider_.get(),
                                                         head);
                }

                Quality evaluateOutOfSample(BiPartition& partition, const CoverageSet& coverageState,
                                            const AbstractPrediction& head) const override {
                    return evaluateOutOfSampleInternally(weights_, coverageState, partition,
                                                         thresholds_.statisticsProvider_.get(), head);
                }

                void recalculatePrediction(const SinglePartition& partition, const CoverageMask& coverageState,
                                           AbstractPrediction& head) const override {
                    recalculatePredictionInternally<SinglePartition::const_iterator>(
                      partition.cbegin(), partition.getNumElements(), coverageState,
                      thresholds_.statisticsProvider_.get(), head);
                }

                void recalculatePrediction(const BiPartition& partition, const CoverageMask& coverageState,
                                           AbstractPrediction& head) const override {
                    recalculatePredictionInternally<BiPartition::const_iterator>(
                      partition.first_cbegin(), partition.getNumFirst(), coverageState,
                      thresholds_.statisticsProvider_.get(), head);
                }

                void recalculatePrediction(const SinglePartition& partition, const CoverageSet& coverageState,
                                           AbstractPrediction& head) const override {
                    recalculatePredictionInternally(coverageState, thresholds_.statisticsProvider_.get(), head);
                }

                void recalculatePrediction(BiPartition& partition, const CoverageSet& coverageState,
                                           AbstractPrediction& head) const override {
                    recalculatePredictionInternally(coverageState, partition, thresholds_.statisticsProvider_.get(),
                                                    head);
                }

                void applyPrediction(const AbstractPrediction& prediction) override {
                    uint32 numCovered = coverageSet_.getNumCovered();
                    CoverageSet::const_iterator iterator = coverageSet_.cbegin();
                    const AbstractPrediction* predictionPtr = &prediction;
                    IStatistics* statisticsPtr = &thresholds_.statisticsProvider_.get();
                    uint32 numThreads = thresholds_.numThreads_;

#pragma omp parallel for firstprivate(numCovered) firstprivate(iterator) firstprivate(predictionPtr) \
  firstprivate(statisticsPtr) schedule(dynamic) num_threads(numThreads)
                    for (int64 i = 0; i < numCovered; i++) {
                        uint32 exampleIndex = iterator[i];
                        predictionPtr->apply(*statisticsPtr, exampleIndex);
                    }
                }

                void revertPrediction(const AbstractPrediction& prediction) override {
                    uint32 numCovered = coverageSet_.getNumCovered();
                    CoverageSet::const_iterator iterator = coverageSet_.cbegin();
                    const AbstractPrediction* predictionPtr = &prediction;
                    IStatistics* statisticsPtr = &thresholds_.statisticsProvider_.get();
                    uint32 numThreads = thresholds_.numThreads_;

#pragma omp parallel for firstprivate(numCovered) firstprivate(iterator) firstprivate(predictionPtr) \
  firstprivate(statisticsPtr) schedule(dynamic) num_threads(numThreads)
                    for (int64 i = 0; i < numCovered; i++) {
                        uint32 exampleIndex = iterator[i];
                        predictionPtr->revert(*statisticsPtr, exampleIndex);
                    }
                }
        };

        const std::unique_ptr<IFeatureBinning> numericalFeatureBinningPtr_;

        const std::unique_ptr<IFeatureBinning> nominalFeatureBinningPtr_;

        const uint32 numThreads_;

        std::unordered_map<uint32, IFeatureBinning::Result> cache_;

    public:

        /**
         * @param featureMatrix                 A reference to an object of type `IColumnWiseFeatureMatrix` that
         *                                      provides column-wise access to the feature values of individual training
         *                                      examples
         * @param featureInfo                   A reference to an object of type `IFeatureInfo` that provides
         *                                      information about the types of individual features
         * @param statisticsProvider            A reference to an object of type `IStatisticsProvider` that provides
         *                                      access to statistics about the labels of the training examples
         * @param numericalFeatureBinningPtr    An unique pointer to an object of type `IFeatureBinning` that should be
         *                                      used to assign numerical feature values to bins
         * @param nominalFeatureBinningPtr      An unique pointer to an object of type `IFeatureBinning` that should be
         *                                      used to assign nominal feature values to bins
         * @param numThreads                    The number of CPU threads to be used to update statistics in parallel
         */
        ApproximateThresholds(const IColumnWiseFeatureMatrix& featureMatrix, const IFeatureInfo& featureInfo,
                              IStatisticsProvider& statisticsProvider,
                              std::unique_ptr<IFeatureBinning> numericalFeatureBinningPtr,
                              std::unique_ptr<IFeatureBinning> nominalFeatureBinningPtr, uint32 numThreads)
            : AbstractThresholds(featureMatrix, featureInfo, statisticsProvider),
              numericalFeatureBinningPtr_(std::move(numericalFeatureBinningPtr)),
              nominalFeatureBinningPtr_(std::move(nominalFeatureBinningPtr)), numThreads_(numThreads) {}

        std::unique_ptr<IThresholdsSubset> createSubset(const EqualWeightVector& weights) override {
            IStatistics& statistics = statisticsProvider_.get();
            std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr = statistics.createWeightedStatistics(weights);
            return std::make_unique<ApproximateThresholds::ThresholdsSubset<EqualWeightVector>>(
              *this, std::move(weightedStatisticsPtr), weights);
        }

        std::unique_ptr<IThresholdsSubset> createSubset(const BitWeightVector& weights) override {
            IStatistics& statistics = statisticsProvider_.get();
            std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr = statistics.createWeightedStatistics(weights);
            return std::make_unique<ApproximateThresholds::ThresholdsSubset<BitWeightVector>>(
              *this, std::move(weightedStatisticsPtr), weights);
        }

        std::unique_ptr<IThresholdsSubset> createSubset(const DenseWeightVector<uint32>& weights) override {
            IStatistics& statistics = statisticsProvider_.get();
            std::unique_ptr<IWeightedStatistics> weightedStatisticsPtr = statistics.createWeightedStatistics(weights);
            return std::make_unique<ApproximateThresholds::ThresholdsSubset<DenseWeightVector<uint32>>>(
              *this, std::move(weightedStatisticsPtr), weights);
        }
};

ApproximateThresholdsFactory::ApproximateThresholdsFactory(
  std::unique_ptr<IFeatureBinningFactory> numericalFeatureBinningFactoryPtr,
  std::unique_ptr<IFeatureBinningFactory> nominalFeatureBinningFactoryPtr, uint32 numThreads)
    : numericalFeatureBinningFactoryPtr_(std::move(numericalFeatureBinningFactoryPtr)),
      nominalFeatureBinningFactoryPtr_(std::move(nominalFeatureBinningFactoryPtr)), numThreads_(numThreads) {}

std::unique_ptr<IThresholds> ApproximateThresholdsFactory::create(const IColumnWiseFeatureMatrix& featureMatrix,
                                                                  const IFeatureInfo& featureInfo,
                                                                  IStatisticsProvider& statisticsProvider) const {
    std::unique_ptr<IFeatureBinning> numericalFeatureBinningPtr = numericalFeatureBinningFactoryPtr_->create();
    std::unique_ptr<IFeatureBinning> nominalFeatureBinningPtr = nominalFeatureBinningFactoryPtr_->create();
    return std::make_unique<ApproximateThresholds>(featureMatrix, featureInfo, statisticsProvider,
                                                   std::move(numericalFeatureBinningPtr),
                                                   std::move(nominalFeatureBinningPtr), numThreads_);
}
