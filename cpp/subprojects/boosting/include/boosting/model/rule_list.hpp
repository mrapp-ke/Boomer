/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/model/model_builder.hpp"


namespace boosting {

    /**
     * Allows to build models that store several rules in the order they have been added.
     */
    class RuleListBuilder final : public IModelBuilder {

        private:

            std::unique_ptr<RuleModel> modelPtr_;

        public:

            RuleListBuilder();

            void setDefaultRule(const AbstractPrediction& prediction) override;

            void addRule(const ConditionList& conditions, const AbstractPrediction& prediction) override;

            std::unique_ptr<RuleModel> build(uint32 numUsedRules) override;

    };

}
