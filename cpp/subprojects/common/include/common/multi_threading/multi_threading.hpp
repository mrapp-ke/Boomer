/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_matrix.hpp"


/**
 * Defines an interface for all classes that allow to configure the multi-threading behavior of a parallelizable
 * algorithm.
 */
class IMultiThreadingConfig {

    public:

        virtual ~IMultiThreadingConfig() { };

        /**
         * Determines and returns the number of threads to be used by a parallelizable algorithm.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param numLabels     The total number of available labels
         * @return              The number of threads to be used
         */
        virtual uint32 getNumThreads(const IFeatureMatrix& featureMatrix, uint32 numLabels) const = 0;

};
