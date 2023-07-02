/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"

#include <vector>

/**
 * A factory that allows to create instances of the type `IStoppingCriterion` that allow to test multiple stopping
 * criteria. If at least one of these stopping criteria is met, the induction of additional rules is stopped.
 */
class StoppingCriterionListFactory final : public IStoppingCriterionFactory {
    private:

        std::vector<std::unique_ptr<IStoppingCriterionFactory>> stoppingCriterionFactories_;

    public:

        /**
         * Adds a new factory that allows to create instances of a stopping criterion to be tested.
         *
         * @param stoppingCriterionFactoryPtr An unique pointer to an object of type `IStoppingCriterionFactory` that
         *                                    should be added
         */
        void addStoppingCriterionFactory(std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr);

        std::unique_ptr<IStoppingCriterion> create(const SinglePartition& partition) const override;

        std::unique_ptr<IStoppingCriterion> create(BiPartition& partition) const override;
};
