#include "seco/heuristics/heuristic_recall.hpp"
#include "heuristic_common.hpp"


namespace seco {

    float64 Recall::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const {
        return recall(cin, crp, uin, urp);
    }

}
