#include "seco/heuristics/heuristic_wra.hpp"
#include "heuristic_common.hpp"


namespace seco {

    float64 WRA::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                         float64 urn, float64 urp) const {
        return wra(cin, cip, crn, crp, uin, uip, urn, urp);
    }

}
