/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/util/quality.hpp"

#include <functional>

/**
 * Defines a function for comparing the quality of different rules.
 */
struct RuleCompareFunction {
    public:

        /**
         * A function for comparing two objects of type `Quality`. It should return true, if the first object is better
         * than the second one, false otherwise.
         */
        typedef std::function<bool(const Quality&, const Quality&)> CompareFunction;

        /**
         * @param c A function of type `CompareFunction` for comparing the quality of different rules
         * @param m The minimum quality of a rule
         */
        RuleCompareFunction(CompareFunction c, float64 m) : compare(c), minQuality(m) {};

        /**
         * A function of type `CompareFunction` for comparing the quality of different rules.
         */
        const CompareFunction compare;

        /**
         * The minimum quality of a rule.
         */
        const float64 minQuality;
};
