/*
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/binning/bin_index_vector.hpp"
#include "common/binning/threshold_vector.hpp"
#include "common/input/feature_matrix.hpp"
#include "common/input/feature_vector.hpp"
#include "common/input/label_matrix.hpp"
#include "common/thresholds/thresholds.hpp"
#include <memory>


/**
 * Defines an interface for methods that assign feature values to bins.
 */
class IFeatureBinning {

    public:

        /**
         * The result that is returned by a binning method. It contains an unique pointer to a vector that stores the
         * thresholds that result from the boundaries of the bins, as well as to a vector that stores the indices of the
         * bins, individual values have been assigned to.
         */
        struct Result {

            /**
             * An unique pointer to an object of type `ThresholdVector` that provides access to the thresholds that
             * result from the boundaries of the bins.
             */
            std::unique_ptr<ThresholdVector> thresholdVectorPtr;

            /**
             * An unique pointer to an object of type `IBinIndexVector` that provides access to the indices of the bins,
             * individual values have been assigned to.
             */
            std::unique_ptr<IBinIndexVector> binIndicesPtr;

        };

        virtual ~IFeatureBinning() { };

        /**
         * Assigns the values in a given `FeatureVector` to bins.
         *
         * @param featureVector A reference to an object of type `FeatureVector` whose values should be assigned to bins
         * @param numExamples   The total number of available training examples
         * @return              An object of type `Result` that contains a vector, which stores thresholds that result
         *                      from the boundaries between the bins, as well as a vector that stores the indices of the
         *                      bins, individual values have been assigned to
         */
        virtual Result createBins(FeatureVector& featureVector, uint32 numExamples) const = 0;

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IFeatureBinning`.
 */
class IFeatureBinningFactory {

    public:

        virtual ~IFeatureBinningFactory() { };

        /**
         * Creates and returns a new object of type `IFeatureBinning`.
         *
         * @return An unique pointer to an object of type `IFeatureBinning` that has been created
         */
        virtual std::unique_ptr<IFeatureBinning> create() const = 0;

};

/**
 * Defines an interface for all classes that allow to configure a method that assigns feature values to bins.
 */
class IFeatureBinningConfig {

    public:

        virtual ~IFeatureBinningConfig() { };

        /**
         * Creates and returns a new object of type `IThresholdsFactory` according to the specified configuration.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param labelMatrix   A reference to an object of type `ILabelMatrix` that provides access to the labels of
         *                      the training examples
         * @return              An unique pointer to an object of type `IThresholdsFactory` that has been created
         */
        virtual std::unique_ptr<IThresholdsFactory> createThresholdsFactory(const IFeatureMatrix& featureMatrix,
                                                                            const ILabelMatrix& labelMatrix) const = 0;

};
