/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_info.hpp"
#include "common/input/feature_matrix_column_wise.hpp"
#include "common/sampling/weight_vector_bit.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/weight_vector_equal.hpp"
#include "common/statistics/statistics_provider.hpp"
#include "common/thresholds/thresholds_subset.hpp"

/**
 * Defines an interface for all classes that provide access to thresholds that may be used by the first condition of a
 * rule that currently has an empty body and therefore covers the entire instance space.
 */
class IThresholds {
    public:

        virtual ~IThresholds() {};

        /**
         * Creates and returns a new subset of the thresholds, which initially contains all of the thresholds.
         *
         * @param weights   A reference to an object of type `EqualWeightVector` that provides access to the weights of
         *                  individual training examples
         * @return          An unique pointer to an object of type `IThresholdsSubset` that has been created
         */
        virtual std::unique_ptr<IThresholdsSubset> createSubset(const EqualWeightVector& weights) = 0;

        /**
         * Creates and returns a new subset of the thresholds, which initially contains all of the thresholds.
         *
         * @param weights   A reference to an object of type `BitWeightVector` that provides access to the weights of
         *                  individual training examples
         * @return          An unique pointer to an object of type `IThresholdsSubset` that has been created
         */
        virtual std::unique_ptr<IThresholdsSubset> createSubset(const BitWeightVector& weights) = 0;

        /**
         * Creates and returns a new subset of the thresholds, which initially contains all of the thresholds.
         *
         * @param weights   A reference to an object of type `DenseWeightVector<uint32>` that provides access to the
         *                  weights of individual training examples
         * @return          An unique pointer to an object of type `IThresholdsSubset` that has been created
         */
        virtual std::unique_ptr<IThresholdsSubset> createSubset(const DenseWeightVector<uint32>& weights) = 0;

        /**
         * Returns a reference to an object of type `IStatisticsProvider` that provides access to the statistics that
         * correspond to individual training examples in the instance space.
         *
         * @return A reference to an object of type `IStatisticsProvider`
         */
        virtual IStatisticsProvider& getStatisticsProvider() const = 0;
};

/**
 * Defines an interface for all classes that allow to create instances of the type `IThresholds`.
 */
class IThresholdsFactory {
    public:

        virtual ~IThresholdsFactory() {};

        /**
         * Creates and returns a new object of type `IThresholds`.
         *
         * @param featureMatrix         A reference to an object of type `IColumnWiseFeatureMatrix` that provides
         *                              column-wise access to the feature values of individual training examples
         * @param featureInfo           A reference  to an object of type `IFeatureInfo` that provides information about
         *                              the types of individual features
         * @param statisticsProvider    A reference to an object of type `IStatisticsProvider` that provides access to
         *                              statistics about the labels of the training examples
         * @return                      An unique pointer to an object of type `IThresholds` that has been created
         */
        virtual std::unique_ptr<IThresholds> create(const IColumnWiseFeatureMatrix& featureMatrix,
                                                    const IFeatureInfo& featureInfo,
                                                    IStatisticsProvider& statisticsProvider) const = 0;
};
