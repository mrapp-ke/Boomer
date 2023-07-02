/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss.hpp"
#include "boosting/rule_evaluation/head_type.hpp"
#include "boosting/statistics/statistic_format.hpp"
#include "common/rule_model_assemblage/default_rule.hpp"

namespace boosting {

    /**
     * Allows to configure a method that automatically decides whether a default rule should be included in a rule-based
     * model or not.
     */
    class AutomaticDefaultRuleConfig final : public IDefaultRuleConfig {
        private:

            const std::unique_ptr<IStatisticsConfig>& statisticsConfigPtr_;

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

            const std::unique_ptr<IHeadConfig>& headConfigPtr_;

        public:

            /**
             * @param statisticsConfigPtr   A reference to an unique pointer that stores the configuration of the
             *                              statistics
             * @param lossConfigPtr         A reference to an unique pointer that stores the configuration of the loss
             *                              function
             * @param headConfigPtr         A reference to an unique pointer that stores the configuration of the rule
             *                              heads
             */
            AutomaticDefaultRuleConfig(const std::unique_ptr<IStatisticsConfig>& statisticsConfigPtr,
                                       const std::unique_ptr<ILossConfig>& lossConfigPtr,
                                       const std::unique_ptr<IHeadConfig>& headConfigPtr);

            /**
             * @see `IDefaultRuleConfig::isDefaultRuleUsed`
             */
            bool isDefaultRuleUsed(const IRowWiseLabelMatrix& labelMatrix) const override;
    };

}
