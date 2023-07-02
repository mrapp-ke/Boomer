#include "common/stopping/stopping_criterion_list.hpp"

/**
 * An implementation of the type `IStoppingCriterion` that tests multiple stopping criteria.
 *
 *
 * @tparam Partition The type of the object that provides access to the indices of the examples that are included in the
 *                   holdout set
 */
template<typename Partition>
class StoppingCriterionList final : public IStoppingCriterion {
    private:

        std::vector<std::unique_ptr<IStoppingCriterion>> stoppingCriteria_;

    public:

        /**
         * @param partition                     A reference to an object of template type `Partition` that provides
         *                                      access to the indices of the examples that are included in the holdout
         *                                      set
         * @param stoppingCriterionFactories    A reference to a vector that stores the factories that allow to create
         *                                      instances of the stopping criteria to be tested
         */
        StoppingCriterionList(
          Partition& partition,
          const std::vector<std::unique_ptr<IStoppingCriterionFactory>>& stoppingCriterionFactories) {
            stoppingCriteria_.reserve(stoppingCriterionFactories.size());

            for (auto it = stoppingCriterionFactories.cbegin(); it != stoppingCriterionFactories.cend(); it++) {
                const std::unique_ptr<IStoppingCriterionFactory>& stoppingCriterionFactoryPtr = *it;
                stoppingCriteria_.push_back(std::move(stoppingCriterionFactoryPtr->create(partition)));
            }
        }

        Result test(const IStatistics& statistics, uint32 numRules) override {
            Result result;

            for (auto it = stoppingCriteria_.begin(); it != stoppingCriteria_.end(); it++) {
                std::unique_ptr<IStoppingCriterion>& stoppingCriterionPtr = *it;
                Result stoppingCriterionResult = stoppingCriterionPtr->test(statistics, numRules);
                result.stop |= stoppingCriterionResult.stop;
                uint32 numUsedRules = stoppingCriterionResult.numUsedRules;

                if (numUsedRules != 0) {
                    result.numUsedRules = numUsedRules;
                }
            }

            return result;
        }
};

/**
 * An implementation of the type `IStoppingCriterion` that does not test for any stopping criteria.
 */
class NoStoppingCriterion final : public IStoppingCriterion {
    public:

        Result test(const IStatistics& statistics, uint32 numRules) override {
            Result result;
            return result;
        }
};

void StoppingCriterionListFactory::addStoppingCriterionFactory(
  std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr) {
    stoppingCriterionFactories_.push_back(std::move(stoppingCriterionFactoryPtr));
}

std::unique_ptr<IStoppingCriterion> StoppingCriterionListFactory::create(const SinglePartition& partition) const {
    if (stoppingCriterionFactories_.empty()) {
        return std::make_unique<NoStoppingCriterion>();
    } else {
        return std::make_unique<StoppingCriterionList<const SinglePartition>>(partition, stoppingCriterionFactories_);
    }
}

std::unique_ptr<IStoppingCriterion> StoppingCriterionListFactory::create(BiPartition& partition) const {
    if (stoppingCriterionFactories_.empty()) {
        return std::make_unique<NoStoppingCriterion>();
    } else {
        return std::make_unique<StoppingCriterionList<BiPartition>>(partition, stoppingCriterionFactories_);
    }
}
