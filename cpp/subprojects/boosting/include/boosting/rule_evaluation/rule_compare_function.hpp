/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_evaluation/rule_compare_function.hpp"

namespace boosting {

    /**
     * Returns whether the quality of a boosted rule is better than the quality of a second one.
     *
     * @param first     An object of type `Quality` that represents the quality of the first rule
     * @param second    An object of type `Quality` that represents the quality of the second rule
     * @return          True, if the quality of the first rule is better than the quality of the second one, false
     *                  otherwise
     */
    static inline constexpr bool compareBoostedRuleQuality(const Quality& first, const Quality& second) {
        return first.quality < second.quality;
    }

    /**
     * An object of type `RuleCompareFunction` that defines the function that should be used for comparing the quality
     * of boosted rules.
     */
    static const RuleCompareFunction BOOSTED_RULE_COMPARE_FUNCTION(compareBoostedRuleQuality, 0.0);

}
