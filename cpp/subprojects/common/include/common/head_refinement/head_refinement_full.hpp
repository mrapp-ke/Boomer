/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/head_refinement/head_refinement_factory.hpp"


/**
 * Allows to create instances of the class `FullHeadRefinement`.
 */
class FullHeadRefinementFactory final : public IHeadRefinementFactory {

    public:

        std::unique_ptr<IHeadRefinement> create(const FullIndexVector& labelIndices) const override;

        std::unique_ptr<IHeadRefinement> create(const PartialIndexVector& labelIndices) const override;

};
