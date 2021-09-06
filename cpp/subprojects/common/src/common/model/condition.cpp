#include "common/model/condition.hpp"

Condition::Condition() {

}

Condition::Condition(const Condition& condition)
    : featureIndex(condition.featureIndex), comparator(condition.comparator), threshold(condition.threshold),
      start(condition.start), end(condition.end), covered(condition.covered), numCovered(condition.numCovered) {

}
