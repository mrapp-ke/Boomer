/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/pruning/pruning.hpp"


/**
 * Implements incremental reduced error pruning (IREP) for pruning classification rules.
 *
 * Given `n` conditions in the order of their induction, IREP allows to remove up to `n - 1` trailing conditions,
 * depending on which of the resulting rules improves the most over the quality score of the original rules as measured
 * on the prune set.
 */
class IREP final : public IPruning {

    public:

        std::unique_ptr<ICoverageState> prune(IThresholdsSubset& thresholdsSubset, IPartition& partition,
                                              ConditionList& conditions, const AbstractPrediction& head) const override;

};
