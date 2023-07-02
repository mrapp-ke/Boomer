#include "boosting/binning/label_binning_no.hpp"

#include "boosting/rule_evaluation/rule_evaluation_example_wise_complete.hpp"
#include "boosting/rule_evaluation/rule_evaluation_example_wise_partial_dynamic.hpp"
#include "boosting/rule_evaluation/rule_evaluation_example_wise_partial_fixed.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_complete.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_dynamic.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_partial_fixed.hpp"

namespace boosting {

    NoLabelBinningConfig::NoLabelBinningConfig(const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
                                               const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr)
        : l1RegularizationConfigPtr_(l1RegularizationConfigPtr), l2RegularizationConfigPtr_(l2RegularizationConfigPtr) {

    }

    std::unique_ptr<ILabelWiseRuleEvaluationFactory>
      NoLabelBinningConfig::createLabelWiseCompleteRuleEvaluationFactory() const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        return std::make_unique<LabelWiseCompleteRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
    }

    std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory>
      NoLabelBinningConfig::createLabelWiseFixedPartialRuleEvaluationFactory(float32 labelRatio, uint32 minLabels,
                                                                             uint32 maxLabels) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        return std::make_unique<LabelWiseFixedPartialRuleEvaluationFactory>(
          labelRatio, minLabels, maxLabels, l1RegularizationWeight, l2RegularizationWeight);
    }

    std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory>
      NoLabelBinningConfig::createLabelWiseDynamicPartialRuleEvaluationFactory(float32 threshold,
                                                                               float32 exponent) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        return std::make_unique<LabelWiseDynamicPartialRuleEvaluationFactory>(
          threshold, exponent, l1RegularizationWeight, l2RegularizationWeight);
    }

    std::unique_ptr<IExampleWiseRuleEvaluationFactory>
      NoLabelBinningConfig::createExampleWiseCompleteRuleEvaluationFactory(const Blas& blas,
                                                                           const Lapack& lapack) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        return std::make_unique<ExampleWiseCompleteRuleEvaluationFactory>(l1RegularizationWeight,
                                                                          l2RegularizationWeight, blas, lapack);
    }

    std::unique_ptr<IExampleWiseRuleEvaluationFactory>
      NoLabelBinningConfig::createExampleWiseFixedPartialRuleEvaluationFactory(float32 labelRatio, uint32 minLabels,
                                                                               uint32 maxLabels, const Blas& blas,
                                                                               const Lapack& lapack) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        return std::make_unique<ExampleWiseFixedPartialRuleEvaluationFactory>(
          labelRatio, minLabels, maxLabels, l1RegularizationWeight, l2RegularizationWeight, blas, lapack);
    }

    std::unique_ptr<IExampleWiseRuleEvaluationFactory>
      NoLabelBinningConfig::createExampleWiseDynamicPartialRuleEvaluationFactory(float32 threshold, float32 exponent,
                                                                                 const Blas& blas,
                                                                                 const Lapack& lapack) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        return std::make_unique<ExampleWiseDynamicPartialRuleEvaluationFactory>(
          threshold, exponent, l1RegularizationWeight, l2RegularizationWeight, blas, lapack);
    }

}
