#include "seco/heuristics/heuristic_m_estimate.hpp"
#include "common/validation.hpp"
#include "heuristic_common.hpp"
#include <cmath>


namespace seco {

    MEstimate::MEstimate(float64 m)
        : m_(m) {
        assertGreaterOrEqual<float64>("m", m, 0);
    }

    float64 MEstimate::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                               float64 uip, float64 urn, float64 urp) const {
        if (std::isinf(m_)) {
            // Equivalent to weighted relative accuracy
            return wra(cin, cip, crn, crp, uin, uip, urn, urp);
        } else if (m_ > 0) {
            // Trade-off between precision and weighted relative accuracy
            float64 numCoveredEqual = cin + crp;
            float64 numCovered = numCoveredEqual + cip + crn;

            if (numCovered == 0) {
                return 1;
            }

            float64 numUncoveredEqual = uin + urp;
            float64 numTotal = numCovered + numUncoveredEqual + uip + urn;
            float64 numEqual = numCoveredEqual + numUncoveredEqual;
            return 1 - ((numCoveredEqual + (m_ * (numEqual / numTotal))) / (numCovered + m_));
        } else {
            // Equivalent to precision
            return precision(cin, cip, crn, crp);
        }
    }

}
