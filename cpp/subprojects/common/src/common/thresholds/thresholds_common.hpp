/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/thresholds/thresholds.hpp"
#include "common/input/feature_matrix.hpp"
#include "common/input/nominal_feature_mask.hpp"
#include "common/head_refinement/head_refinement_factory.hpp"
#include "common/statistics/statistics_provider.hpp"
#include "omp.h"


static inline void updateSampledStatisticsInternally(IStatistics& statistics, const IWeightVector& weights) {
    uint32 numExamples = statistics.getNumStatistics();
    statistics.resetSampledStatistics();

    for (uint32 i = 0; i < numExamples; i++) {
        uint32 weight = weights.getWeight(i);
        statistics.addSampledStatistic(i, weight);
    }
}

template<class T>
static inline float64 evaluateOutOfSampleInternally(T iterator, uint32 numExamples, const IWeightVector& weights,
                                                    const CoverageMask& coverageMask, const IStatistics& statistics,
                                                    const IHeadRefinementFactory& headRefinementFactory,
                                                    const AbstractPrediction& prediction) {
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = prediction.createSubset(statistics);

    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = iterator[i];

        if (weights.getWeight(exampleIndex) == 0 && coverageMask.isCovered(exampleIndex)) {
            statisticsSubsetPtr->addToSubset(exampleIndex, 1);
        }
    }

    std::unique_ptr<IHeadRefinement> headRefinementPtr = prediction.createHeadRefinement(headRefinementFactory);
    const IScoreVector& scoreVector = headRefinementPtr->calculatePrediction(*statisticsSubsetPtr, false, false);
    return scoreVector.overallQualityScore;
}

static inline float64 evaluateOutOfSampleInternally(const IWeightVector& weights, const CoverageSet& coverageSet,
                                                    const IStatistics& statistics,
                                                    const IHeadRefinementFactory& headRefinementFactory,
                                                    const AbstractPrediction& prediction) {
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = prediction.createSubset(statistics);
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator iterator = coverageSet.cbegin();

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = iterator[i];

        if (weights.getWeight(exampleIndex) == 0) {
            statisticsSubsetPtr->addToSubset(exampleIndex, 1);
        }
    }

    std::unique_ptr<IHeadRefinement> headRefinementPtr = prediction.createHeadRefinement(headRefinementFactory);
    const IScoreVector& scoreVector = headRefinementPtr->calculatePrediction(*statisticsSubsetPtr, false, false);
    return scoreVector.overallQualityScore;
}

static inline float64 evaluateOutOfSampleInternally(const IWeightVector& weights, const CoverageSet& coverageSet,
                                                    BiPartition& partition, const IStatistics& statistics,
                                                    const IHeadRefinementFactory& headRefinementFactory,
                                                    const AbstractPrediction& prediction) {
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = prediction.createSubset(statistics);
    const std::unordered_set<uint32>& holdoutSet = partition.getSecondSet();
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator iterator = coverageSet.cbegin();

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = iterator[i];

        if (weights.getWeight(exampleIndex) == 0 && holdoutSet.find(exampleIndex) == holdoutSet.cend()) {
            statisticsSubsetPtr->addToSubset(exampleIndex, 1);
        }
    }

    std::unique_ptr<IHeadRefinement> headRefinementPtr = prediction.createHeadRefinement(headRefinementFactory);
    const IScoreVector& scoreVector = headRefinementPtr->calculatePrediction(*statisticsSubsetPtr, false, false);
    return scoreVector.overallQualityScore;
}

template<class T>
static inline void recalculatePredictionInternally(T iterator, uint32 numExamples, const CoverageMask& coverageMask,
                                                   const IStatistics& statistics,
                                                   const IHeadRefinementFactory& headRefinementFactory,
                                                   Refinement& refinement) {
    AbstractPrediction& head = *refinement.headPtr;
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = head.createSubset(statistics);

    for (uint32 i = 0; i < numExamples; i++) {
        uint32 exampleIndex = iterator[i];

        if (coverageMask.isCovered(exampleIndex)) {
            statisticsSubsetPtr->addToSubset(exampleIndex, 1);
        }
    }

    std::unique_ptr<IHeadRefinement> headRefinementPtr = head.createHeadRefinement(headRefinementFactory);
    const IScoreVector& scoreVector = headRefinementPtr->calculatePrediction(*statisticsSubsetPtr, false, false);
    scoreVector.updatePrediction(head);
}

static inline void recalculatePredictionInternally(const CoverageSet& coverageSet, const IStatistics& statistics,
                                                   const IHeadRefinementFactory& headRefinementFactory,
                                                   Refinement& refinement) {
    AbstractPrediction& head = *refinement.headPtr;
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = head.createSubset(statistics);
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator iterator = coverageSet.cbegin();

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = iterator[i];
        statisticsSubsetPtr->addToSubset(exampleIndex, 1);
    }

    std::unique_ptr<IHeadRefinement> headRefinementPtr = head.createHeadRefinement(headRefinementFactory);
    const IScoreVector& scoreVector = headRefinementPtr->calculatePrediction(*statisticsSubsetPtr, false, false);
    scoreVector.updatePrediction(head);
}

static inline void recalculatePredictionInternally(const CoverageSet& coverageSet, BiPartition& partition,
                                                   const IStatistics& statistics,
                                                   const IHeadRefinementFactory& headRefinementFactory,
                                                   Refinement& refinement) {
    AbstractPrediction& head = *refinement.headPtr;
    std::unique_ptr<IStatisticsSubset> statisticsSubsetPtr = head.createSubset(statistics);
    const std::unordered_set<uint32>& holdoutSet = partition.getSecondSet();
    uint32 numCovered = coverageSet.getNumCovered();
    CoverageSet::const_iterator iterator = coverageSet.cbegin();

    for (uint32 i = 0; i < numCovered; i++) {
        uint32 exampleIndex = iterator[i];

        if (holdoutSet.find(exampleIndex) == holdoutSet.cend()) {
            statisticsSubsetPtr->addToSubset(exampleIndex, 1);
        }
    }

    std::unique_ptr<IHeadRefinement> headRefinementPtr = head.createHeadRefinement(headRefinementFactory);
    const IScoreVector& scoreVector = headRefinementPtr->calculatePrediction(*statisticsSubsetPtr, false, false);
    scoreVector.updatePrediction(head);
}

/**
 * An abstract base class for all classes that provide access to thresholds that may be used by the first condition of a
 * rule that currently has an empty body and therefore covers the entire instance space.
 */
class AbstractThresholds : public IThresholds {

    protected:

        /**
         * A shared pointer to an object of type `IFeatureMatrix` that provides access to the feature values of the
         * training examples.
         */
        std::shared_ptr<IFeatureMatrix> featureMatrixPtr_;

        /**
         * A shared pointer to an object of type `INominalFeatureMask` that provides access to the information whether
         * individual feature are nominal or not.
         */
        std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr_;

        /**
         * A shared pointer to an object of type `IStatisticsProvider` that provides access to statistics about the
         * labels of the training examples.
         */
        std::shared_ptr<IStatisticsProvider> statisticsProviderPtr_;

        /**
         * A shared pointer to an object of type `IHeadRefinementFactory` that allows to create instances of the class
         * that should be used to find the heads of rules.
         */
        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr_;

    public:

        /**
         * @param featureMatrixPtr          A shared pointer to an object of type `IFeatureMatrix` that provides access
         *                                  to the feature values of the training examples
         * @param nominalFeatureMaskPtr     A shared pointer to an object of type `INominalFeatureMask` that provides
         *                                  access to the information whether individual features are nominal or not
         * @param statisticsProviderPtr     A shared pointer to an object of type `IStatisticsProvider` that provides
                                            access to statistics about the labels of the training examples
         * @param headRefinementFactoryPtr  A shared pointer to an object of type `IHeadRefinementFactory` that allows
         *                                  to create instances of the class that should be used to find the heads of
         *                                  rules
         */
        AbstractThresholds(std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                           std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
                           std::shared_ptr<IStatisticsProvider> statisticsProviderPtr,
                           std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr)
            : featureMatrixPtr_(featureMatrixPtr), nominalFeatureMaskPtr_(nominalFeatureMaskPtr),
              statisticsProviderPtr_(statisticsProviderPtr), headRefinementFactoryPtr_(headRefinementFactoryPtr) {

        }

        virtual ~AbstractThresholds() { };

        uint32 getNumExamples() const override final {
            return featureMatrixPtr_->getNumRows();
        }

        uint32 getNumFeatures() const override final {
            return featureMatrixPtr_->getNumCols();
        }

        uint32 getNumLabels() const override final {
            return statisticsProviderPtr_->get().getNumLabels();
        }

};
