/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/head_refinement/prediction.hpp"
#include "common/model/rule_model.hpp"
#include "common/model/condition_list.hpp"


/**
 * Defines an interface for all classes that allow to incrementally build rule-based models.
 */
class IModelBuilder {

    public:

        virtual ~IModelBuilder() { };

        /**
         * Sets the default rule of the model.
         *
         * @param prediction A pointer to an object of type `AbstractPrediction` that stores the scores that are
         *                   predicted by the default rule
         */
        virtual void setDefaultRule(const AbstractPrediction& prediction) = 0;

        /**
         * Adds a new rule to the model.
         *
         * @param conditions    A reference to an object of type `ConditionList` that stores the rule's conditions
         * @param prediction    A reference to an object of type `AbstractPrediction` that stores the scores that are
         *                      predicted by the rule
         */
        virtual void addRule(const ConditionList& conditions, const AbstractPrediction& prediction) = 0;

        /**
         * Builds and returns the model.
         *
         * @param numUsedRules  The number of used rules
         * @return              An unique pointer to an object of type `RuleModel` that has been built
         */
        virtual std::unique_ptr<RuleModel> build(uint32 numUsedRules) = 0;

};
