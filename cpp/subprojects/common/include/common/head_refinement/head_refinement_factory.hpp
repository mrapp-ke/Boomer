/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/head_refinement/head_refinement.hpp"
#include "common/indices/index_vector_full.hpp"
#include "common/indices/index_vector_partial.hpp"


/**
 * Defines an interface for all factories that allow to create instances of the type `IHeadRefinement`.
 */
class IHeadRefinementFactory {

    public:

        virtual ~IHeadRefinementFactory() { };

        /**
         * Creates and returns a new object of type `IHeadRefinement` that allows to find the best head considering all
         * available labels.
         *
         * @param labelIndices  A reference to an object of type `FullIndexVector` that provides access to the indices
         *                      of the labels that should be considered
         * @return              An unique pointer to an object of type `IHeadRefinement` that has been created
         */
        virtual std::unique_ptr<IHeadRefinement> create(const FullIndexVector& labelIndices) const = 0;

        /**
         * Creates and returns a new object of type `IHeadRefinement` that allows to find the best head considering only
         * a subset of the available labels.
         *
         * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
         *                      indices of the labels that should be considered
         * @return              An unique pointer to an object of type `IHeadRefinement` that has been created
         */
        virtual std::unique_ptr<IHeadRefinement> create(const PartialIndexVector& labelIndices) const = 0;

};
