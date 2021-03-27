/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"


/**
 * A stopping criterion that ensures that the number of induced rules does not exceed a certain maximum.
 */
class SizeStoppingCriterion final : public IStoppingCriterion {

    private:

        uint32 maxRules_;

    public:

        /**
         * @param maxRules The maximum number of rules
         */
        SizeStoppingCriterion(uint32 maxRules);

        Result test(const IPartition& partition, const IStatistics& statistics, uint32 numRules) override;

};
