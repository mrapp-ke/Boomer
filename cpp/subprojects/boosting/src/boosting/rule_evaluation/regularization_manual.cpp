#include "boosting/rule_evaluation/regularization_manual.hpp"

#include "common/util/validation.hpp"

#include <iostream>

namespace boosting {

    ManualRegularizationConfig::ManualRegularizationConfig() : regularizationWeight_(1) {}

    float64 ManualRegularizationConfig::getRegularizationWeight() const {
        return regularizationWeight_;
    }

    IManualRegularizationConfig& ManualRegularizationConfig::setRegularizationWeight(float64 regularizationWeight) {
        assertGreater<float64>("regularizationWeight", regularizationWeight, 0);
        regularizationWeight_ = regularizationWeight;
        return *this;
    }

    float64 ManualRegularizationConfig::getWeight() const {
        return regularizationWeight_;
    }

}
