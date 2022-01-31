#include "boosting/multi_threading/parallel_statistic_update_auto.hpp"
#include "boosting/losses/loss_example_wise.hpp"
#include "common/util/threads.hpp"


namespace boosting {

    AutoParallelStatisticUpdateConfig::AutoParallelStatisticUpdateConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr)
        : lossConfigPtr_(lossConfigPtr) {

    }

    uint32 AutoParallelStatisticUpdateConfig::getNumThreads(const IFeatureMatrix& featureMatrix,
                                                            uint32 numLabels) const {
        if (dynamic_cast<const IExampleWiseLossConfig*>(lossConfigPtr_.get()) && numLabels >= 20) {
            return getNumAvailableThreads(0);
        } else {
            return 1;
        }
    };

}
