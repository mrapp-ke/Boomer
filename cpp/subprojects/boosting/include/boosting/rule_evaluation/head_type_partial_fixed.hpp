/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/binning/label_binning.hpp"
#include "boosting/macros.hpp"
#include "boosting/rule_evaluation/head_type.hpp"
#include "boosting/rule_evaluation/regularization.hpp"
#include "common/multi_threading/multi_threading.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure partial rule heads that predict for a predefined
     * number of labels.
     */
    class MLRLBOOSTING_API IFixedPartialHeadConfig {
        public:

            virtual ~IFixedPartialHeadConfig() {};

            /**
             * Returns the percentage that specifies for how many labels the rule heads predict.
             *
             * @return The percentage that specifies for how many labels the rule heads predict or 0, if the percentage
             *         is calculated based on the average label cardinality
             */
            virtual float32 getLabelRatio() const = 0;

            /**
             * Sets the percentage that specifies for how many labels the rule heads should predict.
             *
             * @param labelRatio    A percentage that specifies for how many labels the rule heads should predict, e.g.,
             *                      if 100 labels are available, a percentage of 0.5 means that the rule heads predict
             *                      for a subset of `ceil(0.5 * 100) = 50` labels. Must be in (0, 1) or 0, if the
             *                      percentage should be calculated based on the average label cardinality
             * @return              A reference to an object of type `IFixedPartialHeadConfig` that allows further
             *                      configuration of the rule heads
             */
            virtual IFixedPartialHeadConfig& setLabelRatio(float32 labelRatio) = 0;

            /**
             * Returns the minimum number of labels for which the rule heads predict.
             *
             * @return The minimum number of labels for which the rule heads predict
             */
            virtual uint32 getMinLabels() const = 0;

            /**
             * Sets the minimum number of labels for which the rule heads should predict.
             *
             * @param minLabels The minimum number of labels for which the rule heads should predict. Must be at least 2
             * @return          A reference to an object of type `IFixedPartialHeadConfig` that allows further
             *                  configuration of the rule heads
             */
            virtual IFixedPartialHeadConfig& setMinLabels(uint32 minLabels) = 0;

            /**
             * Returns the maximum number of labels for which the rule heads predict.
             *
             * @return The maximum number of labels for which the rule heads predict
             */
            virtual uint32 getMaxLabels() const = 0;

            /**
             * Sets the maximum number of labels for which the rule heads should predict.
             *
             * @param maxLabels The maximum number of labels for which the rule heads should predict. Must be at least
             *                  the minimum number of labels or 0, if the maximum number of labels should not be
             *                  restricted
             * @return          A reference to an object of type `IFixedPartialHeadConfig` that allows further
             *                  configuration of the rule heads
             */
            virtual IFixedPartialHeadConfig& setMaxLabels(uint32 maxLabels) = 0;
    };

    /**
     * Allows to configure partial rule heads that predict for a predefined number of labels.
     */
    class FixedPartialHeadConfig final : public IHeadConfig,
                                         public IFixedPartialHeadConfig {
        private:

            float32 labelRatio_;

            uint32 minLabels_;

            uint32 maxLabels_;

            const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr_;

            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

        public:

            /**
             * @param labelBinningConfigPtr     A reference to an unique pointer that stores the configuration of the
             *                                  method for assigning labels to bins
             * @param multiThreadingConfigPtr   A reference to an unique pointer that stores the configuration of the
             *                                  multi-threading behavior that should be used for the parallel update of
             *                                  statistics
             */
            FixedPartialHeadConfig(const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr,
                                   const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            float32 getLabelRatio() const override;

            IFixedPartialHeadConfig& setLabelRatio(float32 labelRatio) override;

            uint32 getMinLabels() const override;

            IFixedPartialHeadConfig& setMinLabels(uint32 minLabels) override;

            uint32 getMaxLabels() const override;

            IFixedPartialHeadConfig& setMaxLabels(uint32 maxLabels) override;

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
