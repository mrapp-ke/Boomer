#include "common/pruning/pruning_no.hpp"


std::unique_ptr<ICoverageState> NoPruning::prune(IThresholdsSubset& thresholdsSubset, IPartition& partition,
                                                 ConditionList& conditions, const AbstractPrediction& head) const {
    return nullptr;
}
