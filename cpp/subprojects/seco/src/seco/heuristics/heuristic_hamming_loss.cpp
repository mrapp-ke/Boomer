#include "seco/heuristics/heuristic_hamming_loss.hpp"


namespace seco {

    float64 HammingLoss::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                                 float64 uip, float64 urn, float64 urp) const {
        float64 numCoveredIncorrect = cip + crn;
        float64 numCoveredCorrect = cin + crp;
        float64 numCovered = numCoveredIncorrect + numCoveredCorrect;

        if (numCovered == 0) {
            return 1;
        }

        float64 numIncorrect = numCoveredIncorrect + urn + urp;
        float64 numTotal = numIncorrect + numCoveredCorrect + uin + uip;
        return numIncorrect / numTotal;
    }

}
