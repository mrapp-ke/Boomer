/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/macros.hpp"
#include <memory>

class IClassificationPredictorFactory;
class IClassificationPredictor;
class IRegressionPredictorFactory;
class IRegressionPredictor;
class IProbabilityPredictorFactory;
class IProbabilityPredictor;
class RuleList;


/**
 * Defines an interface for all classes that provide information about the label space that may be used as a basis for
 * making predictions.
 */
class MLRLCOMMON_API ILabelSpaceInfo {

    public:

        virtual ~ILabelSpaceInfo() { };

        /**
         * Creates and returns a new instance of the class `IClassificationPredictor`, based on the type of the label
         * space information.
         *
         * @param factory   A reference to an object of type `IClassificationPredictorFactory` that should be used to
         *                  create the instance
         * @param model     A reference to an object of type `RuleList` that should be used to obtain predictions
         * @return          An unique pointer to an object of type `IClassificationPredictor` that has been created
         */
        virtual std::unique_ptr<IClassificationPredictor> createClassificationPredictor(
            const IClassificationPredictorFactory& factory, const RuleList& model) const = 0;

        /**
         * Creates and returns a new instance of the class `IRegressionPredictor`, based on the type of this rule-based
         * model.
         *
         * @param factory   A reference to an object of type `IRegressionPredictorFactory` that should be used to create
         *                  the instance
         * @param model     A reference to an object of type `RuleList` that should be used to obtain predictions
         * @return          An unique pointer to an object of type `IRegressionPredictor` that has been created
         */
        virtual std::unique_ptr<IRegressionPredictor> createRegressionPredictor(
            const IRegressionPredictorFactory& factory, const RuleList& model) const = 0;

        /**
         * Creates and returns a new instance of the class `IProbabilityPredictor`, based on the type of this rule-based
         * model.
         *
         * @param factory   A reference to an object of type `IProbabilityPredictorFactory` that should be used to
         *                  create the instance
         * @param model     A reference to an object of type `RuleList` that should be used to obtain predictions
         * @return          An unique pointer to an object of type `IProbabilityPredictor` that has been created
         */
        virtual std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
            const IProbabilityPredictorFactory& factory, const RuleList& model) const = 0;

};
