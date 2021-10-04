/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/input/feature_vector.hpp"
#include <memory>


/**
 * Defines an interface for all feature matrices that provide column-wise access to the feature values of the training
 * examples.
 */
class IFeatureMatrix {

    public:

        virtual ~IFeatureMatrix() { };

        /**
         * Returns the number of available examples.
         *
         * @return The number of examples
         */
        virtual uint32 getNumRows() const = 0;

        /**
         * Returns the number of available features.
         *
         * @return The number of features
         */
        virtual uint32 getNumCols() const = 0;

        /**
         * Fetches a feature vector that stores the indices of the training examples, as well as their feature values,
         * for a specific feature and stores it in a given unique pointer.
         *
         * @param featureIndex      The index of the feature
         * @param featureVectorPtr  An unique pointer to an object of type `FeatureVector` that should be used to store
         *                          the feature vector
         */
        virtual void fetchFeatureVector(uint32 featureIndex,
                                        std::unique_ptr<FeatureVector>& featureVectorPtr) const = 0;

};
