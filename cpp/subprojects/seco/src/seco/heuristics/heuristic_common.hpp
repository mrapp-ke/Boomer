/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


namespace seco {

    static inline constexpr float64 precision(float64 cin, float64 cip, float64 crn, float64 crp) {
        float64 numCoveredIncorrect = cip + crn;
        float64 numCovered = numCoveredIncorrect + cin + crp;

        if (numCovered == 0) {
            return 1;
        }

        return numCoveredIncorrect / numCovered;
    }

    static inline constexpr float64 recall(float64 cin, float64 crp, float64 uin, float64 urp) {
        float64 numUncoveredEqual = uin + urp;
        float64 numEqual = numUncoveredEqual + cin + crp;

        if (numEqual == 0) {
            return 1;
        }

        return numUncoveredEqual / numEqual;
    }

    static inline constexpr float64 wra(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp) {
        float64 numCoveredEqual = cin + crp;
        float64 numUncoveredEqual = uin + urp;
        float64 numEqual = numUncoveredEqual + numCoveredEqual;
        float64 numCovered = numCoveredEqual + cip + crn;
        float64 numUncovered = numUncoveredEqual + uip + urn;
        float64 numTotal = numCovered + numUncovered;

        if (numCovered == 0 || numTotal == 0) {
            return 1;
        }

        return 1 - ((numCovered / numTotal) * ((numCoveredEqual / numCovered) - (numEqual / numTotal)));
    }

}
