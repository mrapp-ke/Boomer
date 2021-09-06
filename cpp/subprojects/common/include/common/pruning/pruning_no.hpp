/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/pruning/pruning.hpp"


/**
 * An implementation of the class `IPruning` that does not actually perform any pruning, but retains all conditions.
 */
class NoPruning final : public IPruning {

    public:

        std::unique_ptr<ICoverageState> prune(IThresholdsSubset& thresholdsSubset, IPartition& partition,
                                              ConditionList& conditions, const AbstractPrediction& head) const override;

};
