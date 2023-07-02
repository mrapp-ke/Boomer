/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_info.hpp"
#include "common/input/feature_matrix.hpp"
#include "common/iterator/binary_forward_iterator.hpp"
#include "common/thresholds/thresholds.hpp"
#include "omp.h"

template<typename IndexIterator, typename WeightVector>
static inline Quality evaluateOutOfSampleInternally(IndexIterator indexIterator, uint32 numExamples,
                                                    const WeightVector& weights, const CoverageMask& coverageMask,
                                                    const IStatistics& statistics,
                                                    const AbstractPrediction& prediction) {
    OutOfSampleWeightVector<WeightVector> outOfSampleWeights(weights);
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr =
      prediction.createStatisticsSubset(statistics, outOfSampleWeights);

    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indexIterator[i];

        if (statisticsSubsetPtr->hasNonZeroWeight(exampleIndex) && coverageMask.isCovered(exampleIndex)) {
            statisticsSubsetPtr->addToSubset(exampleIndex);
        }
    }

    return statisticsSubsetPtr->calculateScores();
}

template<typename WeightVector>
static inline Quality evaluateOutOfSampleInternally(const WeightVector& weights, const CoverageSet& coverageSet,
                                                    const IStatistics& statistics,
                                                    const AbstractPrediction& prediction) {
    OutOfSampleWeightVector<WeightVector> outOfSampleWeights(weights);
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr =
      prediction.createStatisticsSubset(statistics, outOfSampleWeights);
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator coverageSetIterator = coverageSet.cbegin();

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = coverageSetIterator[i];

        if (statisticsSubsetPtr->hasNonZeroWeight(exampleIndex)) {
            statisticsSubsetPtr->addToSubset(exampleIndex);
        }
    }

    return statisticsSubsetPtr->calculateScores();
}

template<typename WeightVector>
static inline Quality evaluateOutOfSampleInternally(const WeightVector& weights, const CoverageSet& coverageSet,
                                                    BiPartition& partition, const IStatistics& statistics,
                                                    const AbstractPrediction& prediction) {
    OutOfSampleWeightVector<WeightVector> outOfSampleWeights(weights);
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr =
      prediction.createStatisticsSubset(statistics, outOfSampleWeights);
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator coverageSetIterator = coverageSet.cbegin();
    partition.sortSecond();
    auto holdoutSetIterator = make_binary_forward_iterator(partition.second_cbegin(), partition.second_cend());
    uint32 previousExampleIndex = 0;

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = coverageSetIterator[i];
        std::advance(holdoutSetIterator, exampleIndex - previousExampleIndex);

        if (*holdoutSetIterator && statisticsSubsetPtr->hasNonZeroWeight(exampleIndex)) {
            statisticsSubsetPtr->addToSubset(exampleIndex);
        }

        previousExampleIndex = exampleIndex;
    }

    return statisticsSubsetPtr->calculateScores();
}

template<typename IndexIterator>
static inline void recalculatePredictionInternally(IndexIterator indexIterator, uint32 numExamples,
                                                   const CoverageMask& coverageMask, const IStatistics& statistics,
                                                   AbstractPrediction& prediction) {
    EqualWeightVector weights(numExamples);
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = prediction.createStatisticsSubset(statistics, weights);

    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = indexIterator[i];

        if (coverageMask.isCovered(exampleIndex)) {
            statisticsSubsetPtr->addToSubset(exampleIndex);
        }
    }

    const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScores();
    scoreVector.updatePrediction(prediction);
}

static inline void recalculatePredictionInternally(const CoverageSet& coverageSet, const IStatistics& statistics,
                                                   AbstractPrediction& prediction) {
    uint32 numStatistics = statistics.getNumStatistics();
    EqualWeightVector weights(numStatistics);
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = prediction.createStatisticsSubset(statistics, weights);
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator coverageSetIterator = coverageSet.cbegin();

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = coverageSetIterator[i];
        statisticsSubsetPtr->addToSubset(exampleIndex);
    }

    const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScores();
    scoreVector.updatePrediction(prediction);
}

static inline void recalculatePredictionInternally(const CoverageSet& coverageSet, BiPartition& partition,
                                                   const IStatistics& statistics, AbstractPrediction& prediction) {
    uint32 numStatistics = statistics.getNumStatistics();
    EqualWeightVector weights(numStatistics);
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = prediction.createStatisticsSubset(statistics, weights);
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator coverageSetIterator = coverageSet.cbegin();
    partition.sortSecond();
    auto holdoutSetIterator = make_binary_forward_iterator(partition.second_cbegin(), partition.second_cend());
    uint32 previousExampleIndex = 0;

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = coverageSetIterator[i];
        std::advance(holdoutSetIterator, exampleIndex - previousExampleIndex);

        if (*holdoutSetIterator) {
            statisticsSubsetPtr->addToSubset(exampleIndex);
        }

        previousExampleIndex = exampleIndex;
    }

    const IScoreVector& scoreVector = statisticsSubsetPtr->calculateScores();
    scoreVector.updatePrediction(prediction);
}

/**
 * An abstract base class for all classes that provide access to thresholds that may be used by the first condition of a
 * rule that currently has an empty body and therefore covers the entire instance space.
 */
class AbstractThresholds : public IThresholds {
    protected:

        /**
         * A reference to an object of type `IColumnWiseFeatureMatrix` that provides column-wise access to the feature
         * values of the training examples.
         */
        const IColumnWiseFeatureMatrix& featureMatrix_;

        /**
         * A reference to an object of type `IFeatureInfo` that provides information about the types of individual
         * features.
         */
        const IFeatureInfo& featureInfo_;

        /**
         * A reference to an object of type `IStatisticsProvider` that provides access to statistics about the labels of
         * the training examples.
         */
        IStatisticsProvider& statisticsProvider_;

    public:

        /**
         * @param featureMatrix         A reference to an object of type `IColumnWiseFeatureMatrix` that provides
         *                              column-wise access to the feature values of individual training examples
         * @param featureInfo           A reference  to an object of type `IFeatureInfo` that provides information about
         *                              the types of individual features
         * @param statisticsProvider    A reference to an object of type `IStatisticsProvider` that provides access to
         *                              statistics about the labels of the training examples
         */
        AbstractThresholds(const IColumnWiseFeatureMatrix& featureMatrix, const IFeatureInfo& featureInfo,
                           IStatisticsProvider& statisticsProvider)
            : featureMatrix_(featureMatrix), featureInfo_(featureInfo), statisticsProvider_(statisticsProvider) {}

        virtual ~AbstractThresholds() override {};

        IStatisticsProvider& getStatisticsProvider() const override final {
            return statisticsProvider_;
        }
};
