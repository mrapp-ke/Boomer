/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_refinement/prediction.hpp"

/**
 * Defines an interface for all classes that allow to post-process the predictions of rules once they have been learned.
 */
class IPostProcessor {
    public:

        virtual ~IPostProcessor() {};

        /**
         * Post-processes the prediction of a rule.
         *
         * @param prediction A reference to an object of type `AbstractPrediction` that stores the predictions of a rule
         */
        virtual void postProcess(AbstractPrediction& prediction) const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IPostProcessor`.
 */
class IPostProcessorFactory {
    public:

        virtual ~IPostProcessorFactory() {};

        /**
         * Creates and returns a new object of type `IPostProcessor`.
         *
         * @return An unique pointer to an object of type `IPostProcessor` that has been created
         */
        virtual std::unique_ptr<IPostProcessor> create() const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method that post-processes the predictions of rules
 * once they have been learned.
 */
class IPostProcessorConfig {
    public:

        virtual ~IPostProcessorConfig() {};

        /**
         * Creates and returns a new object of type `IPostProcessorFactory` according to the specified configuration.
         *
         * @return An unique pointer to an object of type `IPostProcessorFactory` that has been created
         */
        virtual std::unique_ptr<IPostProcessorFactory> createPostProcessorFactory() const = 0;
};
