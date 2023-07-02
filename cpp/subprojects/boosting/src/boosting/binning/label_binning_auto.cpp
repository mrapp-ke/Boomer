#include "boosting/binning/label_binning_auto.hpp"

#include "boosting/binning/label_binning_equal_width.hpp"
#include "boosting/binning/label_binning_no.hpp"

namespace boosting {

    AutomaticLabelBinningConfig::AutomaticLabelBinningConfig(
      const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
      const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr)
        : l1RegularizationConfigPtr_(l1RegularizationConfigPtr), l2RegularizationConfigPtr_(l2RegularizationConfigPtr) {

    }

    std::unique_ptr<ILabelWiseRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createLabelWiseCompleteRuleEvaluationFactory() const {
        return NoLabelBinningConfig(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)
          .createLabelWiseCompleteRuleEvaluationFactory();
    }

    std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createLabelWiseFixedPartialRuleEvaluationFactory(float32 labelRatio,
                                                                                    uint32 minLabels,
                                                                                    uint32 maxLabels) const {
        return NoLabelBinningConfig(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)
          .createLabelWiseFixedPartialRuleEvaluationFactory(labelRatio, minLabels, maxLabels);
    }

    std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createLabelWiseDynamicPartialRuleEvaluationFactory(float32 threshold,
                                                                                      float32 exponent) const {
        return NoLabelBinningConfig(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)
          .createLabelWiseDynamicPartialRuleEvaluationFactory(threshold, exponent);
    }

    std::unique_ptr<IExampleWiseRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createExampleWiseCompleteRuleEvaluationFactory(const Blas& blas,
                                                                                  const Lapack& lapack) const {
        return EqualWidthLabelBinningConfig(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)
          .createExampleWiseCompleteRuleEvaluationFactory(blas, lapack);
    }

    std::unique_ptr<IExampleWiseRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createExampleWiseFixedPartialRuleEvaluationFactory(
        float32 labelRatio, uint32 minLabels, uint32 maxLabels, const Blas& blas, const Lapack& lapack) const {
        return EqualWidthLabelBinningConfig(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)
          .createExampleWiseFixedPartialRuleEvaluationFactory(labelRatio, minLabels, maxLabels, blas, lapack);
    }

    std::unique_ptr<IExampleWiseRuleEvaluationFactory>
      AutomaticLabelBinningConfig::createExampleWiseDynamicPartialRuleEvaluationFactory(float32 threshold,
                                                                                        float32 exponent,
                                                                                        const Blas& blas,
                                                                                        const Lapack& lapack) const {
        return EqualWidthLabelBinningConfig(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)
          .createExampleWiseDynamicPartialRuleEvaluationFactory(threshold, exponent, blas, lapack);
    }

}
