#include "boosting/rule_evaluation/regularization_no.hpp"


namespace boosting {

    float64 NoRegularizationConfig::getWeight() const {
        return 0;
    }

}
