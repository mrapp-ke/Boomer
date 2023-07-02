/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/model/condition_list.hpp"
#include "common/model/rule_model.hpp"
#include "common/rule_refinement/prediction_evaluated.hpp"

/**
 * Defines an interface for all classes that allow to incrementally build rule-based models.
 */
class IModelBuilder {
    public:

        virtual ~IModelBuilder() {};

        /**
         * Sets the default rule of the model.
         *
         * @param predictionPtr A reference to an unique pointer of type `AbstractEvaluatedPrediction` that stores the
         *                      scores that are predicted by the default rule
         */
        virtual void setDefaultRule(std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) = 0;

        /**
         * Adds a new rule to the model.
         *
         * @param conditionListPtr  A reference to an unique pointer of type `ConditionList` that stores the rule's
         *                          conditions
         * @param predictionPtr     A reference to an unique pointer of type `AbstractEvaluatedPrediction` that stores
         *                          the scores that are predicted by the rule
         */
        virtual void addRule(std::unique_ptr<ConditionList>& conditionListPtr,
                             std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) = 0;

        /**
         * Sets the number of used rules.
         *
         * @param numUsedRules The number of used rules
         */
        virtual void setNumUsedRules(uint32 numUsedRules) = 0;

        /**
         * Builds and returns the model.
         *
         * @return An unique pointer to an object of type `IRuleModel` that has been built
         */
        virtual std::unique_ptr<IRuleModel> buildModel() = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IModelBuilder`.
 */
class IModelBuilderFactory {
    public:

        virtual ~IModelBuilderFactory() {};

        /**
         * Creates and returns a new instance of type `IModelBuilder`.
         *
         * @return An unique pointer to an object of type `IModelBuilder` that has been created
         */
        virtual std::unique_ptr<IModelBuilder> create() const = 0;
};
