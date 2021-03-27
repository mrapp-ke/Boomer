/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * A heuristic that calculates as the (weighted) harmonic mean between the heuristics `Precision` and `Recall`,
     * where the parameter `beta` allows to trade-off between both heuristics. If `beta = 1`, both heuristics are
     * weighed equally. If `beta = 0`, this heuristic is equivalent to the heuristic `Precision`. As `beta` approaches
     * infinity, this heuristic becomes equivalent to the heuristic `Recall`.
     */
    class FMeasure final : public IHeuristic {

        private:

            /**
             * The value of the beta-parameter.
             */
            float64 beta_;

        public:

            /**
             * Creates a new heuristic that calculates as the (weighted) harmonic mean between the heuristics
             * `Precision` and `Recall`.
             *
             * @param beta The value of the beta-parameter. Must be at least 0
             */
            FMeasure(float64 beta);

            float64 evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const override;

    };

}
