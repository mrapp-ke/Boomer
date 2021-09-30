#include "seco/heuristics/heuristic_laplace.hpp"


namespace seco {

    float64 Laplace::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                             float64 uip, float64 urn, float64 urp) const {
        float64 numCoveredIncorrect = cip + crn;
        float64 numCovered = numCoveredIncorrect + cin + crp;
        return (numCoveredIncorrect + 1) / (numCovered + 2);
    }

}
