/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_matrix.hpp"
#include "common/input/feature_vector.hpp"

#include <memory>

/**
 * Defines an interface for all feature matrices that provide column-wise access to the feature values of examples.
 */
class MLRLCOMMON_API IColumnWiseFeatureMatrix : virtual public IFeatureMatrix {
    public:

        virtual ~IColumnWiseFeatureMatrix() override {};

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
