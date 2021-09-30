/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/model/model_builder.hpp"


namespace seco {

    /**
     * Allows to build models that store several rules in the order they have been added, except for the default rule, which
     * is always located at the end.
     */
    class DecisionListBuilder final : public IModelBuilder {

        private:

            std::unique_ptr<IHead> defaultHeadPtr_;

            std::unique_ptr<RuleModel> modelPtr_;

        public:

            DecisionListBuilder();

            void setDefaultRule(const AbstractPrediction& prediction) override;

            void addRule(const ConditionList& conditions, const AbstractPrediction& prediction) override;

            std::unique_ptr<RuleModel> build(uint32 numUsedRules) override;

    };

}
