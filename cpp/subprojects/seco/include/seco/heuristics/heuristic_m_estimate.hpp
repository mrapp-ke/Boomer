/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * A heuristic that allows to trade off between the heuristics `Precision` and `WRA`. The `m`-parameter allows to
     * control the trade-off between both heuristics. If `m = 0`, this heuristic is equivalent to the heuristic
     * `Precision`. As `m` approaches infinity, the isometrics of this heuristic become equivalent to those of the
     * heuristic `WRA`.
     */
    class MEstimate final : public IHeuristic {

        private:

            /**
             * The value of the m-parameter.
             */
            float64 m_;

        public:

            /**
             * Creates a new heuristic that allows to trade off between the heuristics `Precision` and `WRA`.
             *
             * @param m The value of the m-parameter. Must be at least 0
             */
            MEstimate(float64 m);

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override;

    };

}
