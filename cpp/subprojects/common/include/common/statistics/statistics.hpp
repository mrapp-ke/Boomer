/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_refinement/prediction_complete.hpp"
#include "common/rule_refinement/prediction_partial.hpp"
#include "common/sampling/weight_vector_bit.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/weight_vector_equal.hpp"
#include "common/sampling/weight_vector_out_of_sample.hpp"
#include "common/statistics/statistics_weighted.hpp"

/**
 * Defines an interface for all classes that provide access to statistics about the labels of the training examples,
 * which serve as the basis for learning a new rule or refining an existing one.
 */
class IStatistics {
    public:

        virtual ~IStatistics() {};

        /**
         * Returns the number of available statistics.
         *
         * @return The number of statistics
         */
        virtual uint32 getNumStatistics() const = 0;

        /**
         * Returns the number of available labels.
         *
         * @return The number of labels
         */
        virtual uint32 getNumLabels() const = 0;

        /**
         * Updates a specific statistic based on the prediction of a rule that predicts for all available labels.
         *
         * This function must be called for each statistic that is covered by the new rule before learning the next
         * rule.
         *
         * @param statisticIndex    The index of the statistic to be updated
         * @param prediction        A reference to an object of type `CompletePrediction` that stores the scores that
         *                          are predicted by the rule
         */
        virtual void applyPrediction(uint32 statisticIndex, const CompletePrediction& prediction) = 0;

        /**
         * Updates a specific statistic based on the prediction of a rule that predicts for a subset of the available
         * labels.
         *
         * This function must be called for each statistic that is covered by the new rule before learning the next
         * rule.
         *
         * @param statisticIndex    The index of the statistic to be updated
         * @param prediction        A reference to an object of type `PartialPrediction` that stores the scores that are
         *                          predicted by the rule
         */
        virtual void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) = 0;

        /**
         * Reverts a specific statistic that has previously been updated via the function `applyPrediction` based on the
         * prediction of a rule that predicts for all available labels.
         *
         * @param statisticIndex    The index of the statistic to be reverted
         * @param prediction        A reference to an object of type `CompletePrediction` that stores the scores that
         *                          are predicted by the rule
         */
        virtual void revertPrediction(uint32 statisticIndex, const CompletePrediction& prediction) = 0;

        /**
         * Reverts a specific statistic that has previously been updated via the function `applyPrediction` based on the
         * prediction of a rule that predicts for a subset of the available labels.
         *
         * @param statisticIndex    The index of the statistic to be reverted
         * @param prediction        A reference to an object of type `PartialPrediction` that stores the scores that are
         *                          predicted by the rule
         */
        virtual void revertPrediction(uint32 statisticIndex, const PartialPrediction& prediction) = 0;

        /**
         * Calculates and returns a numerical score that assesses the quality of the current predictions for a specific
         * statistic.
         *
         * @param statisticIndex    The index of the statistic for which the predictions should be evaluated
         * @return                  The numerical score that has been calculated
         */
        virtual float64 evaluatePrediction(uint32 statisticIndex) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those labels, whose indices
         * are provided by a specific `CompleteIndexVector`.
         *
         * @param labelIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @param weights       A reference to an object of type `EqualWeightVector` that provides access to the weights
         *                      of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& labelIndices,
                                                                const EqualWeightVector& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those labels, whose indices
         * are provided by a specific `PartialIndexVector`.
         *
         * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @param weights       A reference to an object of type `EqualWeightVector` that provides access to the weights
         *                      of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices,
                                                                const EqualWeightVector& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those labels, whose indices
         * are provided by a specific `CompleteIndexVector`.
         *
         * @param labelIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @param weights       A reference to an object of type `BitWeightVector` that provides access to the weights
         *                      of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& labelIndices,
                                                                const BitWeightVector& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those labels, whose indices
         * are provided by a specific `PartialIndexVector`.
         *
         * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @param weights       A reference to an object of type `BitWeightVector` that provides access to the weights
         *                      of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices,
                                                                const BitWeightVector& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those labels, whose indices
         * are provided by a specific `CompleteIndexVector`.
         *
         * @param labelIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @param weights       A reference to an object of type `DenseWeightVector<uint32>` that provides access to the
         *                      weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const CompleteIndexVector& labelIndices,
                                                                const DenseWeightVector<uint32>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those labels, whose indices
         * are provided by a specific `PartialIndexVector`.
         *
         * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @param weights       A reference to an object of type `DenseWeightVector<uint32>` that provides access to the
         *                      weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(const PartialIndexVector& labelIndices,
                                                                const DenseWeightVector<uint32>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those labels, whose indices
         * are provided by a specific `CompleteIndexVector`.
         *
         * @param labelIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<EqualWeightVector>` that
         *                      provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(
          const CompleteIndexVector& labelIndices, const OutOfSampleWeightVector<EqualWeightVector>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those labels, whose indices
         * are provided by a specific `PartialIndexVector`.
         *
         * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<EqualWeightVector>` that
         *                      provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(
          const PartialIndexVector& labelIndices, const OutOfSampleWeightVector<EqualWeightVector>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those labels, whose indices
         * are provided by a specific `CompleteIndexVector`.
         *
         * @param labelIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<BitWeightVector>` that
         *                      provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(
          const CompleteIndexVector& labelIndices, const OutOfSampleWeightVector<BitWeightVector>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those labels, whose indices
         * are provided by a specific `PartialIndexVector`.
         *
         * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<BitWeightVector>` that
         *                      provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(
          const PartialIndexVector& labelIndices, const OutOfSampleWeightVector<BitWeightVector>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those labels, whose indices
         * are provided by a specific `CompleteIndexVector`.
         *
         * @param labelIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<DenseWeightVector<uint32>>`
         *                      that provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(
          const CompleteIndexVector& labelIndices,
          const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IStatisticsSubset` that includes only those labels, whose indices
         * are provided by a specific `PartialIndexVector`.
         *
         * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the labels that should be included in the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<DenseWeightVector<uint32>>`
         *                      that provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createSubset(
          const PartialIndexVector& labelIndices,
          const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const = 0;

        /**
         * Creates and returns a new object of type `IWeightedStatistics`.
         *
         * @param weights   A reference to an object of type `EqualWeightVector` that provides access to the weights of
         *                  individual training examples
         * @return          An unique pointer to an object of type `IWeightedStatistics` that has been created
         */
        virtual std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
          const EqualWeightVector& weights) const = 0;

        /**
         * Creates and returns a new object of type `IWeightedStatistics`.
         *
         * @param weights   A reference to an object of type `BitWeightVector` that provides access to the weights of
         *                  individual training examples
         * @return          An unique pointer to an object of type `IWeightedStatistics` that has been created
         */
        virtual std::unique_ptr<IWeightedStatistics> createWeightedStatistics(const BitWeightVector& weights) const = 0;

        /**
         * Creates and returns a new object of type `IWeightedStatistics`.
         *
         * @param weights   A reference to an object of type `DenseWeightVector<uint32>` that provides access to the
         *                  weights of individual training examples
         * @return          An unique pointer to an object of type `IWeightedStatistics` that has been created
         */
        virtual std::unique_ptr<IWeightedStatistics> createWeightedStatistics(
          const DenseWeightVector<uint32>& weights) const = 0;
};
