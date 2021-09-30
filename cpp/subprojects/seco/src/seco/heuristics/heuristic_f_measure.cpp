#include "seco/heuristics/heuristic_f_measure.hpp"
#include "common/validation.hpp"
#include "heuristic_common.hpp"
#include <cmath>


namespace seco {

    FMeasure::FMeasure(float64 beta)
        : beta_(beta) {
        assertGreaterOrEqual<float64>("beta", beta, 0);
    }

    float64 FMeasure::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                              float64 uip, float64 urn, float64 urp) const {
        if (std::isinf(beta_)) {
            // Equivalent to recall
            return recall(cin, crp, uin, urp);
        } else if (beta_ > 0) {
            // Weighted harmonic mean between precision and recall
            float64 numCoveredEqual = cin + crp;
            float64 betaPow = beta_ * beta_;
            float64 numerator = (1 + betaPow) * numCoveredEqual;
            float64 denominator = numerator + (betaPow * (uin + urp)) + (cip + crn);

            if (denominator == 0) {
                return 1;
            }

            return 1 - (numerator / denominator);
        } else {
            // Equivalent to precision
            return precision(cin, cip, crn, crp);
        }
    }

}
