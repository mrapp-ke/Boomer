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
     * Defines an interface for all classes that allow to configure partial rule heads that predict for a subset of the
     * available labels that is determined dynamically. Only those labels for which the square of the predictive quality
     * exceeds a certain threshold are included in a rule head.
     */
    class MLRLBOOSTING_API IDynamicPartialHeadConfig {
        public:

            virtual ~IDynamicPartialHeadConfig() {};

            /**
             * Returns the threshold that affects for how many labels the rule heads predict.
             *
             * @return The threshold that affects for how many labels the rule heads predict
             */
            virtual float32 getThreshold() const = 0;

            /**
             * Sets the threshold that affects for how many labels the rule heads should predict.
             *
             * @param threshold A threshold that affects for how many labels the rule heads should predict. A smaller
             *                  threshold results in less labels being selected. A greater threshold results in more
             *                  labels being selected. E.g., a threshold of 0.2 means that a rule will only predict for
             *                  a label if the estimated predictive quality `q` for this particular label satisfies the
             *                  inequality `q^exponent > q_best^exponent * (1 - 0.2)`, where `q_best` is the best
             *                  quality among all labels. Must be in (0, 1)
             * @return          A reference to an object of type `IDynamicPartialHeadConfig` that allows further
             *                  configuration of the rule heads
             */
            virtual IDynamicPartialHeadConfig& setThreshold(float32 threshold) = 0;

            /**
             * Sets the exponent that is used to weigh the estimated predictive quality for individual labels.
             *
             * @return The exponent that is used to weight the estimated predictive quality for individual labels
             */
            virtual float32 getExponent() const = 0;

            /**
             * Sets the exponent that should be used to weigh the estimated predictive quality for individual labels.
             *
             * @param exponent  An exponent that should be used to weigh the estimated predictive quality for individual
             *                  labels. E.g., an exponent of 2 means that the estimated predictive quality `q` for a
             *                  particular label is weighed as `q^2`. Must be at least 1
             * @return          A reference to an object of type `IDynamicPartialHeadConfig` that allows further
             *                  configuration of the rule heads
             */
            virtual IDynamicPartialHeadConfig& setExponent(float32 exponent) = 0;
    };

    /**
     * Allows to configure partial rule heads that predict for a for a subset of the available labels that is determined
     * dynamically. Only those labels for which the square of the predictive quality exceeds a certain threshold are
     * included in a rule head.
     */
    class DynamicPartialHeadConfig final : public IHeadConfig,
                                           public IDynamicPartialHeadConfig {
        private:

            float32 threshold_;

            float32 exponent_;

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
            DynamicPartialHeadConfig(const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr,
                                     const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            float32 getThreshold() const override;

            IDynamicPartialHeadConfig& setThreshold(float32 threshold) override;

            float32 getExponent() const override;

            IDynamicPartialHeadConfig& setExponent(float32 exponent) override;

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
