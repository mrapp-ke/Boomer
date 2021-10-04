/*
 * @author Andreas Seidl Fernandez (aseidlfernandez@gmail.com)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * A heuristic that measures the fraction of incorrectly predicted labels among all labels, i.e., in contrast to the
     * precision metric, examples that are not covered by a rule are taken into account as well.
     *
     * This heuristic is used in the pruning phase of IREP ("Incremental Reduced Error Pruning", Fürnkranz, Widmer 1994,
     * see https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.43.7813&rep=rep1&type=pdf).
     */
    class Accuracy final : public IHeuristic {

        public:

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override;

    };

}
