#include "common/stopping/stopping_criterion_size.hpp"


SizeStoppingCriterion::SizeStoppingCriterion(uint32 maxRules)
    : maxRules_(maxRules) {

}

IStoppingCriterion::Result SizeStoppingCriterion::test(const IPartition& partition, const IStatistics& statistics,
                                                       uint32 numRules) {
    Result result;

    if (numRules < maxRules_) {
        result.action = CONTINUE;
    } else {
        result.action = FORCE_STOP;
        result.numRules = numRules;
    }

    return result;
}
