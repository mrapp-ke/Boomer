/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss_example_wise.hpp"
#include "boosting/rule_evaluation/head_type.hpp"


namespace boosting {

    /**
     * Allows to configure a loss function that implements a multi-label variant of the logistic loss that is applied
     * example-wise.
     */
    class ExampleWiseLogisticLossConfig final : public IExampleWiseLossConfig {

        private:

            const std::unique_ptr<IHeadConfig>& headConfigPtr_;

        public:

            /**
             * @param headConfigPtr A reference to an unique pointer that stores the configuration of rules heads
             */
            ExampleWiseLogisticLossConfig(const std::unique_ptr<IHeadConfig>& headConfigPtr);

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix, const Blas& blas,
                const Lapack& lapack) const override;

            std::unique_ptr<IProbabilityFunctionFactory> createProbabilityFunctionFactory() const override;

            float64 getDefaultPrediction() const override;

            std::unique_ptr<IExampleWiseLossFactory> createExampleWiseLossFactory() const override;


    };

}
