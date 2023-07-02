/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/model/model_builder.hpp"

namespace boosting {

    /**
     * Allows to create instances of the type `IModelBuilder` that build models that store several rules in the order
     * they have been added.
     */
    class RuleListBuilderFactory final : public IModelBuilderFactory {
        public:

            /**
             * @see `IModelBuilderFactory::create`
             */
            std::unique_ptr<IModelBuilder> create() const override;
    };

}
