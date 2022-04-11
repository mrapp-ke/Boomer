#include "boosting/binning/label_binning_no.hpp"
#include "boosting/rule_evaluation/rule_evaluation_example_wise_complete.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_complete.hpp"


namespace boosting {

    NoLabelBinningConfig::NoLabelBinningConfig(const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
                                               const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr)
        : l1RegularizationConfigPtr_(l1RegularizationConfigPtr), l2RegularizationConfigPtr_(l2RegularizationConfigPtr) {

    }

    std::unique_ptr<ILabelWiseRuleEvaluationFactory> NoLabelBinningConfig::createLabelWiseRuleEvaluationFactory() const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        return std::make_unique<LabelWiseCompleteRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
    }

    std::unique_ptr<IExampleWiseRuleEvaluationFactory> NoLabelBinningConfig::createExampleWiseRuleEvaluationFactory(
            const Blas& blas, const Lapack& lapack) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        return std::make_unique<ExampleWiseCompleteRuleEvaluationFactory>(
            l1RegularizationWeight, l2RegularizationWeight, blas, lapack);
    }

}
