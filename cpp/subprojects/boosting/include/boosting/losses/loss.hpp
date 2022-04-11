/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_matrix.hpp"
#include "common/input/label_matrix.hpp"
#include "common/measures/measure_evaluation.hpp"
#include "common/measures/measure_similarity.hpp"
#include "common/statistics/statistics_provider.hpp"
#include "boosting/math/blas.hpp"
#include "boosting/math/lapack.hpp"
#include "boosting/output/probability_function.hpp"


namespace boosting {

    /**
     * Defines an interface for all loss functions.
     */
    class ILoss : public IEvaluationMeasure, public ISimilarityMeasure {

        public:

            virtual ~ILoss() override { };

    };

    /**
     * Defines an interface for all classes that allow to configure a loss function.
     */
    class ILossConfig {

        public:

            virtual ~ILossConfig() { };

            /**
             * Creates and returns a new object of type `IStatisticsProviderFactory` according to the specified
             * configuration.
             *
             * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the
             *                      feature values of the training examples
             * @param labelMatrix   A reference to an object of type `ILabelMatrix` that provides access to the labels
             *                      of the training examples
             * @param blas          A reference to an object of type `Blas` that allows to execute BLAS routines
             * @param lapack        A reference to an object of type `Lapack` that allows to execute LAPACK routines
             * @return              An unique pointer to an object of type `IStatisticsProviderFactory` that has been
             *                      created
             */
            virtual std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix, const Blas& blas,
                const Lapack& lapack) const = 0;

            /**
             * Creates and returns a new object of type `IEvaluationMeasureFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `IEvaluationMeasureFactory` that has been created
             */
            virtual std::unique_ptr<IEvaluationMeasureFactory> createEvaluationMeasureFactory() const = 0;

            /**
             * Creates and returns a new object of type `ISimilarityMeasureFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `ISimilarityMeasureFactory` that has been created
             */
            virtual std::unique_ptr<ISimilarityMeasureFactory> createSimilarityMeasureFactory() const = 0;

            /**
             * Creates and returns a new object of type `IProbabilityFunctionFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `IProbabilityFunctionFactory` that has been created or a
             *         null pointer, if the loss function does not support the prediction of probabilities
             */
            virtual std::unique_ptr<IProbabilityFunctionFactory> createProbabilityFunctionFactory() const = 0;

            /**
             * Returns the default prediction for an example that is not covered by any rules.
             *
             * @return The default prediction
             */
            virtual float64 getDefaultPrediction() const = 0;

    };

};
