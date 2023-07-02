/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/vector_binned_dense.hpp"
#include "common/data/vector_dense.hpp"
#include "common/indices/index_vector.hpp"
#include "common/sampling/weight_vector_bit.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/weight_vector_equal.hpp"
#include "common/sampling/weight_vector_out_of_sample.hpp"

#include <memory>

// Forward declarations
class IStatistics;
class IStatisticsSubset;
class IHead;

/**
 * An abstract base class for all classes that store the scores that are predicted by a rule.
 */
class AbstractPrediction : public IIndexVector {
    protected:

        /**
         * A vector that stores the predicted scores.
         */
        DenseVector<float64> predictedScoreVector_;

    public:

        /**
         * @param numElements The number of labels for which the rule predicts
         */
        AbstractPrediction(uint32 numElements);

        /**
         * An iterator that provides access to the predicted scores and allows to modify them.
         */
        typedef DenseVector<float64>::iterator score_iterator;

        /**
         * An iterator that provides read-only access to the predicted scores.
         */
        typedef DenseVector<float64>::const_iterator score_const_iterator;

        /**
         * Returns a `score_iterator` to the beginning of the predicted scores.
         *
         * @return A `score_iterator` to the beginning
         */
        score_iterator scores_begin();

        /**
         * Returns a `score_iterator` to the end of the predicted scores.
         *
         * @return A `score_iterator` to the end
         */
        score_iterator scores_end();

        /**
         * Returns a `score_const_iterator` to the beginning of the predicted scores.
         *
         * @return A `score_const_iterator` to the beginning
         */
        score_const_iterator scores_cbegin() const;

        /**
         * Returns a `score_const_iterator` to the end of the predicted scores.
         *
         * @return A `score_const_iterator` to the end
         */
        score_const_iterator scores_cend() const;

        /**
         * Sets the predicted scores in another vector to this vector.
         *
         * @param begin A `score_const_iterator` to the beginning of the predicted scores
         * @param end   A `score_const_iterator` to the end of the predicted scores
         */
        void set(score_const_iterator begin, score_const_iterator end);

        /**
         * Sets the predicted scores in another vector to this vector.
         *
         * @param begin An iterator to the beginning of the predicted scores
         * @param end   An iterator to the end of the predicted scores
         */
        void set(DenseBinnedVector<float64>::const_iterator begin, DenseBinnedVector<float64>::const_iterator end);

        /**
         * Updates the given statistics by applying this prediction.
         *
         * @param statistics        A reference to an object of type `IStatistics` to be updated
         * @param statisticIndex    The index of the statistic to be updated
         */
        virtual void apply(IStatistics& statistics, uint32 statisticIndex) const = 0;

        /**
         * Updates the given statistics by reverting this prediction.
         *
         * @param statistics        A reference to an object of type `IStatistics` to be updated
         * @param statisticIndex    The index of the statistic to be updated
         */
        virtual void revert(IStatistics& statistics, uint32 statisticIndex) const = 0;

        /**
         * Sorts the scores that stored by this prediction in increasing order by the the indices of the labels they
         * correspond to.
         */
        virtual void sort() = 0;

        /**
         * Creates and returns a head that contains the scores that are stored by this prediction.
         *
         * @return An unique pointer to an object of type `IHead` that has been created
         */
        virtual std::unique_ptr<IHead> createHead() const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the labels whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `IStatistics` that should be used to create the subset
         * @param weights       A reference to an object of type `EqualWeightVector` that provides access to the weights
         *                      of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createStatisticsSubset(const IStatistics& statistics,
                                                                          const EqualWeightVector& weights) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the labels whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `IStatistics` that should be used to create the subset
         * @param weights       A reference to an object of type `BitWeightVector` that provides access to the weights
         *                      of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createStatisticsSubset(const IStatistics& statistics,
                                                                          const BitWeightVector& weights) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the labels whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `IStatistics` that should be used to create the subset
         * @param weights       A reference to an object of type `DenseWeightVector<uint32>` that provides access to the
         *                      weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const DenseWeightVector<uint32>& weights) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the labels whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `IStatistics` that should be used to create the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<EqualWeightVector>` that
         *                      provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const OutOfSampleWeightVector<EqualWeightVector>& weights) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the labels whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `IStatistics` that should be used to create the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<BitWeightVector>` that
         *                      provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const OutOfSampleWeightVector<BitWeightVector>& weights) const = 0;

        /**
         * Creates and returns a new subset of the given statistics that only contains the labels whose indices are
         * stored in this vector.
         *
         * @param statistics    A reference to an object of type `IStatistics` that should be used to create the subset
         * @param weights       A reference to an object of type `OutOfSampleWeightVector<DenseWeightVector<uint32>>`
         *                      that provides access to the weights of individual training examples
         * @return              An unique pointer to an object of type `IStatisticsSubset` that has been created
         */
        virtual std::unique_ptr<IStatisticsSubset> createStatisticsSubset(
          const IStatistics& statistics, const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const = 0;

        uint32 getNumElements() const override;
};
