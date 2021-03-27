/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/head_refinement/head_refinement_factory.hpp"
#include "seco/head_refinement/lift_function.hpp"


namespace seco {

    /**
     * Allows to create instances of the class `PartialHeadRefinement`.
     */
    class PartialHeadRefinementFactory final : public IHeadRefinementFactory {

        private:

            std::shared_ptr<ILiftFunction> liftFunctionPtr_;

        public:

            /**
             * @param liftFunctionPtr A shared pointer to an object of type `ILiftFunction` that should affect the
             *                        quality scores of rules, depending on how many labels they predict
             */
            PartialHeadRefinementFactory(std::shared_ptr<ILiftFunction> liftFunctionPtr);

            std::unique_ptr<IHeadRefinement> create(const FullIndexVector& labelIndices) const override;

            std::unique_ptr<IHeadRefinement> create(const PartialIndexVector& labelIndices) const override;

    };

}
