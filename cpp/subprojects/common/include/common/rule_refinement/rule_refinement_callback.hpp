/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

/**
 * Defines an interface for callbacks that may be invoked by subclasses of the the class `IRuleRefinement` in order to
 * retrieve the information that is required to search for potential refinements. It consists of statistics, as well as
 * a vector that allows to determine the thresholds that may be used by potential conditions.
 *
 * @tparam Statistics   The type of the statistics,
 * @tparam Vector       The type of the vector that is returned by the callback
 */
template<typename Statistics, typename Vector>
class IRuleRefinementCallback {
    public:

        /**
         * The data that is provided via the callback's `get` function.
         */
        struct Result final {
            public:

                /**
                 * @param s A reference to an object of template type `Statistics` that should be used to search for
                 *          potential refinements
                 * @param v A reference to an object of template type `Vector` that should be used to search for
                 * potential refinements
                 */
                Result(const Statistics& s, const Vector& v) : statistics(s), vector(v) {}

                /**
                 * A reference to an object of template type `Statistics` that should be used to search for potential
                 * refinements.
                 */
                const Statistics& statistics;

                /**
                 * A reference to an object of template type `Vector` that should be used to search for potential
                 * refinements.
                 */
                const Vector& vector;
        };

        virtual ~IRuleRefinementCallback() {};

        /**
         * Invokes the callback and returns its result.
         *
         * @return An object of type `Result` that stores references to the statistics and the vector that may be used
         *         to search for potential refinements
         */
        virtual Result get() = 0;
};
