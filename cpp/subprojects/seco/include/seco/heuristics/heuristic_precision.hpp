/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * A heuristic that measures the fraction of incorrectly predicted labels among all covered labels.
     *
     * This heuristic is equivalent to the pruning heuristic used by RIPPER ("Fast Effective Rule Induction", Cohen
     * 1995, see https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.294.7522&rep=rep1&type=pdf). A proof is
     * provided in the paper "Roc 'n' Rule Learning — Towards a Better Understanding of Covering Algorithms", Fürnkranz,
     * Flach 2005 (see https://link.springer.com/content/pdf/10.1007/s10994-005-5011-x.pdf).
     */
    class Precision final : public IHeuristic {

        public:

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override;

    };

}
