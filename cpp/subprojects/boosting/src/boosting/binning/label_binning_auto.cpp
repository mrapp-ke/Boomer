#include "boosting/binning/label_binning_auto.hpp"
#include "boosting/binning/label_binning_equal_width.hpp"
#include "boosting/binning/label_binning_no.hpp"


namespace boosting {

    AutomaticLabelBinningConfig::AutomaticLabelBinningConfig(
            const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
            const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr)
        : l1RegularizationConfigPtr_(l1RegularizationConfigPtr), l2RegularizationConfigPtr_(l2RegularizationConfigPtr) {

    }

    std::unique_ptr<ILabelWiseRuleEvaluationFactory> AutomaticLabelBinningConfig::createLabelWiseRuleEvaluationFactory() const {
        return NoLabelBinningConfig(l1RegularizationConfigPtr_, l2RegularizationConfigPtr_)
            .createLabelWiseRuleEvaluationFactory();
    }

    std::unique_ptr<IExampleWiseRuleEvaluationFactory> AutomaticLabelBinningConfig::createExampleWiseRuleEvaluationFactory(
            const Blas& blas, const Lapack& lapack) const {
        return EqualWidthLabelBinningConfig(l1RegularizationConfigPtr_,l2RegularizationConfigPtr_)
            .createExampleWiseRuleEvaluationFactory(blas, lapack);
    }

}
