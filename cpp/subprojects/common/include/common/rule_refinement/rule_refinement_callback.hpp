/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics_immutable.hpp"
#include <memory>


/**
 * Defines an interface for callbacks that may be invoked by subclasses of the the class `IRuleRefinement` in order to
 * retrieve the data, consisting of statistics, a vector, as well as corresponding weights, that is required to search
 * for potential refinements.
 *
 * @tparam Vector       The type of the vector that is returned by the callback
 * @tparam WeightVector The type of the weight vector that is returned by the callback
 */
template<class Vector, class WeightVector>
class IRuleRefinementCallback {

    public:

        /**
         * The data that is provided via the callback's `get` function.
         */
        class Result final {

            public:

                /**
                 * @param statistics        A reference to an object of type `IImmutableStatistics` that should be used
                 *                          to search for potential refinements
                 * @param weights           A reference to an object of template type `WeightVector` that provides
                 *                          access to the weights of the elements in `vector`
                 * @param vector            A reference to an object of template type `Vector` that should be used to
                 *                          search for potential refinements
                 */
                Result(const IImmutableStatistics& statistics, const WeightVector& weights, const Vector& vector)
                    : statistics_(statistics), weights_(weights), vector_(vector) {

                }

                /**
                 * A reference to an object of type `IImmutableStatistics` that should be used to search for potential
                 * refinements.
                 */
                const IImmutableStatistics& statistics_;

                /**
                 * A reference to an object of type `WeightVector` that provides access to the weights of the elements
                 * in `vector_`.
                 */
                const WeightVector& weights_;

                /**
                 * A reference to an object of template type `Vector` that should be used to search for potential
                 * refinements.
                 */
                const Vector& vector_;

        };

        virtual ~IRuleRefinementCallback() { };

        /**
         * Invokes the callback and returns its result.
         *
         * @return An unique pointer to an object of type `Result` that stores references to the statistics and the
         *         vector that may be used to search for potential refinements
         */
        virtual std::unique_ptr<Result> get() = 0;

};
