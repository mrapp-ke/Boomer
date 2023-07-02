/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/binning/label_binning.hpp"
#include "boosting/rule_evaluation/head_type.hpp"
#include "boosting/rule_evaluation/regularization.hpp"
#include "common/multi_threading/multi_threading.hpp"

namespace boosting {

    /**
     * Allows to configure complete rule heads that predict for all available labels.
     */
    class CompleteHeadConfig final : public IHeadConfig {
        private:

            const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr_;

            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

            const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr_;

            const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr_;

        public:

            /**
             * @param labelBinningConfigPtr     A reference to an unique pointer that stores the configuration of the
             *                                  method for assigning labels to bins
             * @param multiThreadingConfigPtr   A reference to an unique pointer that stores the configuration of the
             *                                  multi-threading behavior that should be used for the parallel update of
             *                                  statistics
             * @param l1RegularizationConfigPtr A reference to an unique pointer that stores the configuration of the L1
             *                                  regularization
             * @param l2RegularizationConfigPtr A reference to an unique pointer that stores the configuration of the L2
             *                                  regularization
             */
            CompleteHeadConfig(const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr,
                               const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr,
                               const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
                               const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr);

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const ILabelWiseLossConfig& lossConfig) const override;

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const ISparseLabelWiseLossConfig& lossConfig) const override;

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const IExampleWiseLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const override;

            bool isPartial() const override;

            bool isSingleLabel() const override;
    };

}
