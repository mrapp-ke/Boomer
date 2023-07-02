/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss.hpp"
#include "boosting/statistics/statistic_format.hpp"

namespace boosting {

    /**
     * Allows to configure a dense format for storing statistics about the labels of the training examples.
     */
    class DenseStatisticsConfig final : public IStatisticsConfig {
        private:

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

        public:

            /**
             * @param lossConfigPtr A reference to an unique pointer that stores the configuration of the loss function
             */
            DenseStatisticsConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr);

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
              const Lapack& lapack) const override;

            bool isDense() const override;

            bool isSparse() const override;
    };

};
