/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_matrix.hpp"
#include <memory>

// Forward declarations
template<typename T> class DensePredictionMatrix;
class BinarySparsePredictionMatrix;
class IClassificationPredictor;
class IRegressionPredictor;
class IProbabilityPredictor;


/**
 * Defines an interface for all feature matrices that provide row-wise access to the feature values of examples.
 */
class MLRLCOMMON_API IRowWiseFeatureMatrix : virtual public IFeatureMatrix {

    public:

        virtual ~IRowWiseFeatureMatrix() override { };

        /**
         * Obtains and returns dense predictions for all examples in this feature matrix, using a specific
         * `IClassificationPredictor`, depending on the type of this feature matrix.
         *
         * @param predictor A reference to an object of type `IClassificationPredictor` that should be used to obtain
         *                  predictions
         * @param numLabels The number of labels to predict for
         * @return          An unique pointer to an object of type `DensePredictionMatrix` that stores the predictions
         */
        virtual std::unique_ptr<DensePredictionMatrix<uint8>> predictLabels(
            const IClassificationPredictor& predictor, uint32 numLabels) const = 0;

        /**
         * Obtains and returns sparse predictions for all examples in this feature matrix, using a specific
         * `IClassificationPredictor`, depending on the type of this feature matrix.
         *
         * @param predictor A reference to an object of type `IClassificationPredictor` that should be used to obtain
         *                  predictions
         * @param numLabels The number of labels to predict for
         * @return          An unique pointer to an object of type `BinarySparsePredictionMatrix` that stores the
         *                  predictions
         */
        virtual std::unique_ptr<BinarySparsePredictionMatrix> predictSparseLabels(
            const IClassificationPredictor& predictor, uint32 numLabels) const = 0;

        /**
         * Obtains and returns regression scores for all examples in this feature matrix, using a specific
         * `IRegressionPredictor`, depending on the type of this feature matrix.
         *
         * @param predictor A reference to an object of type `IRegressionPredictor` that should be used to obtain
         *                  predictions
         * @param numLabels The number of labels to predict for
         * @return          An unique pointer to an object of type `DensePredictionMatrix` that stores the predictions
         */
        virtual std::unique_ptr<DensePredictionMatrix<float64>> predictScores(
            const IRegressionPredictor& predictor, uint32 numLabels) const = 0;

        /**
         * Obtains and returns probability estimates for all examples in this feature matrix, using a specific
         * `IProbabilityPredictor`, depending on the type of this feature matrix.
         *
         * @param predictor A reference to an object of type `IProbabilityPredictor` that should be used to obtain
         *                  predictions
         * @param numLabels The number of labels to predict for
         * @return          An unique pointer to an object of type `DensePredictionMatrix` that stores the predictions
         */
        virtual std::unique_ptr<DensePredictionMatrix<float64>> predictProbabilities(
            const IProbabilityPredictor& predictor, uint32 numLabels) const = 0;

};
