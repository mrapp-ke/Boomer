/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/post_optimization/post_optimization.hpp"

#include <vector>

/**
 * A factory that allows to create instances of the type `IPostOptimization` that carries out multiple optimization
 * phases.
 */
class PostOptimizationPhaseListFactory final : public IPostOptimizationFactory {
    private:

        std::vector<std::unique_ptr<IPostOptimizationPhaseFactory>> postOptimizationPhaseFactories_;

    public:

        /**
         * Adds a new factory that allows to creates instances of an optimization phase to be carried out.
         *
         * @param postOptimizationPhaseFactoryPtr An unique pointer to an object of type `IPostOptimizationPhaseFactory`
         *                                        that should be added
         */
        void addPostOptimizationPhaseFactory(
          std::unique_ptr<IPostOptimizationPhaseFactory> postOptimizationPhaseFactoryPtr);

        std::unique_ptr<IPostOptimization> create(const IModelBuilderFactory& modelBuilderFactory) const override;
};
