#include "common/pruning/pruning_no.hpp"


/**
 * An implementation of the class `IPruning` that does not actually perform any pruning.
 */
class NoPruning final : public IPruning {

    public:

        std::unique_ptr<ICoverageState> prune(IThresholdsSubset& thresholdsSubset, IPartition& partition,
                                              ConditionList& conditions,
                                              const AbstractPrediction& head) const override {
            return nullptr;
        }

};

/**
 * Allows to create instances of the type `IPruning` that do not actually perform any pruning.
 */
class NoPruningFactory final : public IPruningFactory {

    public:

        std::unique_ptr<IPruning> create() const override {
            return std::make_unique<NoPruning>();
        }

};


std::unique_ptr<IPruningFactory> NoPruningConfig::createPruningFactory() const {
    return std::make_unique<NoPruningFactory>();
}
