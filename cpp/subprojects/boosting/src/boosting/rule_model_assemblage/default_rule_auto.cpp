#include "boosting/rule_model_assemblage/default_rule_auto.hpp"

namespace boosting {

    AutomaticDefaultRuleConfig::AutomaticDefaultRuleConfig(
      const std::unique_ptr<IStatisticsConfig>& statisticsConfigPtr, const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IHeadConfig>& headConfigPtr)
        : statisticsConfigPtr_(statisticsConfigPtr), lossConfigPtr_(lossConfigPtr), headConfigPtr_(headConfigPtr) {}

    bool AutomaticDefaultRuleConfig::isDefaultRuleUsed(const IRowWiseLabelMatrix& labelMatrix) const {
        if (statisticsConfigPtr_->isDense()) {
            return true;
        } else if (statisticsConfigPtr_->isSparse()) {
            return !lossConfigPtr_->isSparse();
        } else {
            return !lossConfigPtr_->isSparse()
                   || !shouldSparseStatisticsBePreferred(labelMatrix, false, headConfigPtr_->isPartial());
        }
    }

}
