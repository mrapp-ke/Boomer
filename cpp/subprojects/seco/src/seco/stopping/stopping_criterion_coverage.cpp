#include "seco/stopping/stopping_criterion_coverage.hpp"
#include "seco/statistics/statistics.hpp"
#include "common/validation.hpp"


namespace seco {

    CoverageStoppingCriterion::CoverageStoppingCriterion(float64 threshold)
        : threshold_(threshold) {
        assertGreaterOrEqual<float64>("threshold", threshold, 0);
    }

    IStoppingCriterion::Result CoverageStoppingCriterion::test(const IPartition& partition,
                                                               const IStatistics& statistics, uint32 numRules) {
        Result result;
        const ICoverageStatistics& coverageStatistics = static_cast<const ICoverageStatistics&>(statistics);

        if (coverageStatistics.getSumOfUncoveredWeights() > threshold_) {
            result.action = CONTINUE;
        } else {
            result.action = FORCE_STOP;
            result.numRules = numRules;
        }

        return result;
    }

}
