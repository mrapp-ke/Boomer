/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
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

            /**
             * @see `IModelBuilder::setDefaultRule`
             */
            void setDefaultRule(const AbstractPrediction& prediction) override;

            /**
             * @see `IModelBuilder::addRule`
             */
            void addRule(const ConditionList& conditions, const AbstractPrediction& prediction) override;

            /**
             * @see `IModelBuilder::build`
             */
            std::unique_ptr<RuleModel> build(uint32 numUsedRules) override;

    };

}
