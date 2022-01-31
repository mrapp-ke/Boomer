/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/post_processing/post_processor.hpp"
#include "boosting/macros.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure a post-processor that shrinks the weights of rules
     * by a constant "shrinkage" parameter.
     */
    class MLRLBOOSTING_API IConstantShrinkageConfig {

        public:

            virtual ~IConstantShrinkageConfig() { };

            /**
             * Returns the value of the "shrinkage" parameter.
             *
             * @return The value of the "shrinkage" parameter
             */
            virtual float64 getShrinkage() const = 0;

            /**
             * Sets the value of the "shrinkage" parameter.
             *
             * @param shrinkage The value of the "shrinkage" parameter. Must be in (0, 1)
             * @return          A reference to an object of type `IConstantShrinkageConfig` that allows further
             *                  configuration of the post-processor
             */
            virtual IConstantShrinkageConfig& setShrinkage(float64 shrinkage) = 0;

    };

    /**
     * Allows to configure a post-processor that shrinks the weights of rules by a constant "shrinkage" parameter.
     */
    class ConstantShrinkageConfig final : public IPostProcessorConfig, public IConstantShrinkageConfig {

        private:

            float64 shrinkage_;

        public:

            ConstantShrinkageConfig();

            float64 getShrinkage() const override;

            IConstantShrinkageConfig& setShrinkage(float64 shrinkage) override;

            /**
             * @see `IPostProcessorConfig::createPostProcessorFactory`
             */
            std::unique_ptr<IPostProcessorFactory> createPostProcessorFactory() const override;

    };

}
