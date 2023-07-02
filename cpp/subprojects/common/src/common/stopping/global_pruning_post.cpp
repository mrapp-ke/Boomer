#include "common/stopping/global_pruning_post.hpp"

#include "common/util/validation.hpp"
#include "global_pruning_common.hpp"

/**
 * An implementation of the type `IStoppingCriterion` that that keeps track of the number of rules in a model that
 * perform best with respect to the examples in the training or holdout set according to a certain measure.
 *
 * @tparam Partition The type of the object that provides access to the indices of the examples that are included in the
 *                   training and holdout set, respectively
 */
template<typename Partition>
class PostPruning final : public IStoppingCriterion {
    private:

        const Partition& partition_;

        const bool useHoldoutSet_;

        const uint32 minRules_;

        const uint32 interval_;

        float64 bestScore_;

        uint32 bestNumRules_;

    public:

        /**
         * @param partition     A reference to an object of template type `Partition` that provides access to the
         *                      indices of the examples that are included in the training and holdout set, respectively
         * @param useHoldoutSet True, if the quality of the current model's predictions should be measured on the
         *                      holdout set, if available, false otherwise
         * @param minRules      The minimum number of rules that must be included in a model. Must be at least 1
         * @param interval      The interval to be used to check whether the current model is the best one evaluated so
         *                      far, e.g., a value of 10 means that the best model may contain 10, 20, ... rules
         */
        PostPruning(const Partition& partition, bool useHoldoutSet, uint32 minRules, uint32 interval)
            : partition_(partition), useHoldoutSet_(useHoldoutSet), minRules_(minRules), interval_(interval),
              bestScore_(std::numeric_limits<float64>::infinity()), bestNumRules_(minRules) {}

        Result test(const IStatistics& statistics, uint32 numRules) override {
            Result result;

            if (numRules >= minRules_ && numRules % interval_ == 0) {
                float64 currentScore = evaluate(partition_, useHoldoutSet_, statistics);

                if (currentScore < bestScore_) {
                    bestScore_ = currentScore;
                    bestNumRules_ = numRules;
                    result.numUsedRules = numRules;
                }
            }

            return result;
        }
};

/**
 * Allows to create implementations of the type `IStoppingCriterion` that stop the induction of rules as soon as the
 * quality of a model's predictions for the examples in the training or holdout set do not improve according a certain
 * measure.
 */
class PostPruningFactory final : public IStoppingCriterionFactory {
    private:

        const bool useHoldoutSet_;

        const uint32 minRules_;

        const uint32 interval_;

    public:

        /**
         * @param useHoldoutSet True, if the quality of the current model's predictions should be measured on the
         *                      holdout set, if available, false otherwise
         * @param minRules      The minimum number of rules that must be included in a model. Must be at least 1
         * @param interval      The interval to be used to check whether the current model is the best one evaluated so
         *                      far, e.g., a value of 10 means that the best model may contain 10, 20, ... rules
         */
        PostPruningFactory(bool useHoldoutSet, uint32 minRules, uint32 interval)
            : useHoldoutSet_(useHoldoutSet), minRules_(minRules), interval_(interval) {}

        std::unique_ptr<IStoppingCriterion> create(const SinglePartition& partition) const override {
            return std::make_unique<PostPruning<const SinglePartition>>(partition, useHoldoutSet_, minRules_,
                                                                        interval_);
        }

        std::unique_ptr<IStoppingCriterion> create(BiPartition& partition) const override {
            return std::make_unique<PostPruning<BiPartition>>(partition, useHoldoutSet_, minRules_, interval_);
        }
};

PostPruningConfig::PostPruningConfig() : useHoldoutSet_(true), removeUnusedRules_(true), minRules_(100), interval_(1) {}

bool PostPruningConfig::isHoldoutSetUsed() const {
    return useHoldoutSet_;
}

IPostPruningConfig& PostPruningConfig::setUseHoldoutSet(bool useHoldoutSet) {
    useHoldoutSet_ = useHoldoutSet;
    return *this;
}

bool PostPruningConfig::isRemoveUnusedRules() const {
    return removeUnusedRules_;
}

IPostPruningConfig& PostPruningConfig::setRemoveUnusedRules(bool removeUnusedRules) {
    removeUnusedRules_ = removeUnusedRules;
    return *this;
}

uint32 PostPruningConfig::getMinRules() const {
    return minRules_;
}

IPostPruningConfig& PostPruningConfig::setMinRules(uint32 minRules) {
    assertGreaterOrEqual<uint32>("minRules", minRules, 1);
    minRules_ = minRules;
    return *this;
}

uint32 PostPruningConfig::getInterval() const {
    return interval_;
}

IPostPruningConfig& PostPruningConfig::setInterval(uint32 interval) {
    assertGreaterOrEqual<uint32>("interval", interval, 1);
    interval_ = interval;
    return *this;
}

std::unique_ptr<IStoppingCriterionFactory> PostPruningConfig::createStoppingCriterionFactory() const {
    return std::make_unique<PostPruningFactory>(useHoldoutSet_, minRules_, interval_);
}

bool PostPruningConfig::shouldUseHoldoutSet() const {
    return useHoldoutSet_;
}

bool PostPruningConfig::shouldRemoveUnusedRules() const {
    return removeUnusedRules_;
}
