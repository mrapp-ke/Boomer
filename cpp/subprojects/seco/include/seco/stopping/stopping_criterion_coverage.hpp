/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/stopping/stopping_criterion.hpp"


namespace seco {

    /**
     * A stopping criterion that stops when the sum of the weights of the uncovered labels provided by
     * `ICoverageStatistics` is smaller than or equal to a certain threshold.
     */
    class CoverageStoppingCriterion final : public IStoppingCriterion {

        private:

            float64 threshold_;

        public:

            /**
             * @param threshold The threshold. Must be at least 0
             */
            CoverageStoppingCriterion(float64 threshold);

            Result test(const IPartition& partition, const IStatistics& statistics, uint32 numRules) override;

    };

}
