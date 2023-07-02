#include "boosting/binning/label_binning_equal_width.hpp"

#include "boosting/rule_evaluation/rule_evaluation_example_wise_complete_binned.hpp"
#include "boosting/rule_evaluation/rule_evaluation_example_wise_partial_dynamic_binned.hpp"
#include "boosting/rule_evaluation/rule_evaluation_example_wise_partial_fixed_binned.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_complete_binned.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_dynamic_binned.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_fixed_binned.hpp"
#include "common/math/math.hpp"
#include "common/util/validation.hpp"

namespace boosting {

    /**
     * Assigns labels to bins, based on the corresponding gradients and Hessians, in a way such that each bin contains
     * labels for which the predicted score is expected to belong to the same value range.
     */
    class EqualWidthLabelBinning final : public ILabelBinning {
        private:

            const float32 binRatio_;

            const uint32 minBins_;

            const uint32 maxBins_;

        public:

            /**
             * @param binRatio  A percentage that specifies how many bins should be used to assign labels to, e.g., if
             *                  100 labels are available, 0.5 means that `ceil(0.5 * 100) = 50` bins should be used.
             *                  Must be in (0, 1)
             * @param minBins   The minimum number of bins to be used to assign labels to. Must be at least 2
             * @param maxBins   The maximum number of bins to be used to assign labels to. Must be at least `minBins` or
             *                  0, if the maximum number of bins should not be restricted
             */
            EqualWidthLabelBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
                : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {}

            uint32 getMaxBins(uint32 numLabels) const override {
                return calculateBoundedFraction(numLabels, binRatio_, minBins_, maxBins_) + 1;
            }

            LabelInfo getLabelInfo(const float64* criteria, uint32 numElements) const override {
                LabelInfo labelInfo;
                labelInfo.numNegativeBins = 0;
                labelInfo.numPositiveBins = 0;

                if (numElements > 0) {
                    labelInfo.minNegative = 0;
                    labelInfo.maxNegative = -std::numeric_limits<float64>::infinity();
                    labelInfo.minPositive = std::numeric_limits<float64>::infinity();
                    labelInfo.maxPositive = 0;

                    for (uint32 i = 0; i < numElements; i++) {
                        float64 criterion = criteria[i];

                        if (criterion < 0) {
                            labelInfo.numNegativeBins++;

                            if (criterion < labelInfo.minNegative) {
                                labelInfo.minNegative = criterion;
                            }

                            if (criterion > labelInfo.maxNegative) {
                                labelInfo.maxNegative = criterion;
                            }
                        } else if (criterion > 0) {
                            labelInfo.numPositiveBins++;

                            if (criterion < labelInfo.minPositive) {
                                labelInfo.minPositive = criterion;
                            }

                            if (criterion > labelInfo.maxPositive) {
                                labelInfo.maxPositive = criterion;
                            }
                        }
                    }

                    if (labelInfo.numNegativeBins > 0) {
                        labelInfo.numNegativeBins =
                          calculateBoundedFraction(labelInfo.numNegativeBins, binRatio_, minBins_, maxBins_);
                    }

                    if (labelInfo.numPositiveBins > 0) {
                        labelInfo.numPositiveBins =
                          calculateBoundedFraction(labelInfo.numPositiveBins, binRatio_, minBins_, maxBins_);
                    }
                }

                return labelInfo;
            }

            void createBins(LabelInfo labelInfo, const float64* criteria, uint32 numElements, Callback callback,
                            ZeroCallback zeroCallback) const override {
                uint32 numNegativeBins = labelInfo.numNegativeBins;
                float64 minNegative = labelInfo.minNegative;
                float64 maxNegative = labelInfo.maxNegative;
                uint32 numPositiveBins = labelInfo.numPositiveBins;
                float64 minPositive = labelInfo.minPositive;
                float64 maxPositive = labelInfo.maxPositive;

                float64 spanPerNegativeBin = minNegative < 0 ? (maxNegative - minNegative) / numNegativeBins : 0;
                float64 spanPerPositiveBin = maxPositive > 0 ? (maxPositive - minPositive) / numPositiveBins : 0;

                for (uint32 i = 0; i < numElements; i++) {
                    float64 criterion = criteria[i];

                    if (criterion < 0) {
                        uint32 binIndex = (uint32) std::floor((criterion - minNegative) / spanPerNegativeBin);

                        if (binIndex >= numNegativeBins) {
                            binIndex = numNegativeBins - 1;
                        }

                        callback(binIndex, i);
                    } else if (criterion > 0) {
                        uint32 binIndex = (uint32) std::floor((criterion - minPositive) / spanPerPositiveBin);

                        if (binIndex >= numPositiveBins) {
                            binIndex = numPositiveBins - 1;
                        }

                        callback(numNegativeBins + binIndex, i);
                    } else {
                        zeroCallback(i);
                    }
                }
            }
    };

    /**
     * Allows to create instances of the class `EqualWidthLabelBinning` that assign labels to bins in a way such that
     * each bin contains labels for which the predicted score is expected to belong to the same value range.
     */
    class EqualWidthLabelBinningFactory final : public ILabelBinningFactory {
        private:

            const float32 binRatio_;

            const uint32 minBins_;

            const uint32 maxBins_;

        public:

            /**
             * @param binRatio  A percentage that specifies how many bins should be used, e.g., if 100 labels are a
             *                  available, a percentage of 0.5 means that `ceil(0.5 * 100) = 50` bins should be used.
             *                  Must be in (0, 1)
             * @param minBins   The minimum number of bins that should be used. Must be at least 2
             * @param maxBins   The maximum number of bins that should be used. Must be at least `minBins` or 0, if the
             *                  maximum number of bins should not be restricted
             */
            EqualWidthLabelBinningFactory(float32 binRatio, uint32 minBins, uint32 maxBins)
                : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {}

            std::unique_ptr<ILabelBinning> create() const override {
                return std::make_unique<EqualWidthLabelBinning>(binRatio_, minBins_, maxBins_);
            }
    };

    EqualWidthLabelBinningConfig::EqualWidthLabelBinningConfig(
      const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
      const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr)
        : binRatio_(0.04f), minBins_(1), maxBins_(0), l1RegularizationConfigPtr_(l1RegularizationConfigPtr),
          l2RegularizationConfigPtr_(l2RegularizationConfigPtr) {}

    float32 EqualWidthLabelBinningConfig::getBinRatio() const {
        return binRatio_;
    }

    IEqualWidthLabelBinningConfig& EqualWidthLabelBinningConfig::setBinRatio(float32 binRatio) {
        assertGreater<float32>("binRatio", binRatio, 0);
        assertLess<float32>("binRatio", binRatio, 1);
        binRatio_ = binRatio;
        return *this;
    }

    uint32 EqualWidthLabelBinningConfig::getMinBins() const {
        return minBins_;
    }

    IEqualWidthLabelBinningConfig& EqualWidthLabelBinningConfig::setMinBins(uint32 minBins) {
        assertGreaterOrEqual<uint32>("minBins", minBins, 1);
        minBins_ = minBins;
        return *this;
    }

    uint32 EqualWidthLabelBinningConfig::getMaxBins() const {
        return maxBins_;
    }

    IEqualWidthLabelBinningConfig& EqualWidthLabelBinningConfig::setMaxBins(uint32 maxBins) {
        if (maxBins != 0) assertGreaterOrEqual<uint32>("maxBins", maxBins, minBins_);
        maxBins_ = maxBins;
        return *this;
    }

    std::unique_ptr<ILabelWiseRuleEvaluationFactory>
      EqualWidthLabelBinningConfig::createLabelWiseCompleteRuleEvaluationFactory() const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr =
          std::make_unique<EqualWidthLabelBinningFactory>(binRatio_, minBins_, maxBins_);
        return std::make_unique<LabelWiseCompleteBinnedRuleEvaluationFactory>(
          l1RegularizationWeight, l2RegularizationWeight, std::move(labelBinningFactoryPtr));
    }

    std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory>
      EqualWidthLabelBinningConfig::createLabelWiseFixedPartialRuleEvaluationFactory(float32 labelRatio,
                                                                                     uint32 minLabels,
                                                                                     uint32 maxLabels) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr =
          std::make_unique<EqualWidthLabelBinningFactory>(binRatio_, minBins_, maxBins_);
        return std::make_unique<LabelWiseFixedPartialBinnedRuleEvaluationFactory>(
          labelRatio, minLabels, maxLabels, l1RegularizationWeight, l2RegularizationWeight,
          std::move(labelBinningFactoryPtr));
    }

    std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory>
      EqualWidthLabelBinningConfig::createLabelWiseDynamicPartialRuleEvaluationFactory(float32 threshold,
                                                                                       float32 exponent) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr =
          std::make_unique<EqualWidthLabelBinningFactory>(binRatio_, minBins_, maxBins_);
        return std::make_unique<LabelWiseDynamicPartialBinnedRuleEvaluationFactory>(
          threshold, exponent, l1RegularizationWeight, l2RegularizationWeight, std::move(labelBinningFactoryPtr));
    }

    std::unique_ptr<IExampleWiseRuleEvaluationFactory>
      EqualWidthLabelBinningConfig::createExampleWiseCompleteRuleEvaluationFactory(const Blas& blas,
                                                                                   const Lapack& lapack) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr =
          std::make_unique<EqualWidthLabelBinningFactory>(binRatio_, minBins_, maxBins_);
        return std::make_unique<ExampleWiseCompleteBinnedRuleEvaluationFactory>(
          l1RegularizationWeight, l2RegularizationWeight, std::move(labelBinningFactoryPtr), blas, lapack);
    }

    std::unique_ptr<IExampleWiseRuleEvaluationFactory>
      EqualWidthLabelBinningConfig::createExampleWiseFixedPartialRuleEvaluationFactory(
        float32 labelRatio, uint32 minLabels, uint32 maxLabels, const Blas& blas, const Lapack& lapack) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr =
          std::make_unique<EqualWidthLabelBinningFactory>(binRatio_, minBins_, maxBins_);
        return std::make_unique<ExampleWiseFixedPartialBinnedRuleEvaluationFactory>(
          labelRatio, minLabels, maxLabels, l1RegularizationWeight, l2RegularizationWeight,
          std::move(labelBinningFactoryPtr), blas, lapack);
    }

    std::unique_ptr<IExampleWiseRuleEvaluationFactory>
      EqualWidthLabelBinningConfig::createExampleWiseDynamicPartialRuleEvaluationFactory(float32 threshold,
                                                                                         float32 exponent,
                                                                                         const Blas& blas,
                                                                                         const Lapack& lapack) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr =
          std::make_unique<EqualWidthLabelBinningFactory>(binRatio_, minBins_, maxBins_);
        return std::make_unique<ExampleWiseDynamicPartialBinnedRuleEvaluationFactory>(
          threshold, exponent, l1RegularizationWeight, l2RegularizationWeight, std::move(labelBinningFactoryPtr), blas,
          lapack);
    }

}
