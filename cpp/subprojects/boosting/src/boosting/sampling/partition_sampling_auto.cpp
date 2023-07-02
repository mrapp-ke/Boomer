#include "boosting/sampling/partition_sampling_auto.hpp"

#include "common/sampling/partition_sampling_bi_stratified_label_wise.hpp"
#include "common/sampling/partition_sampling_no.hpp"

namespace boosting {

    AutomaticPartitionSamplingConfig::AutomaticPartitionSamplingConfig(
      const std::unique_ptr<IGlobalPruningConfig>& globalPruningConfigPtr,
      const std::unique_ptr<IMarginalProbabilityCalibratorConfig>& marginalProbabilityCalibratorConfigPtr,
      const std::unique_ptr<IJointProbabilityCalibratorConfig>& jointProbabilityCalibratorConfigPtr)
        : globalPruningConfigPtr_(globalPruningConfigPtr),
          marginalProbabilityCalibratorConfigPtr_(marginalProbabilityCalibratorConfigPtr),
          jointProbabilityCalibratorConfigPtr_(jointProbabilityCalibratorConfigPtr) {}

    std::unique_ptr<IPartitionSamplingFactory> AutomaticPartitionSamplingConfig::createPartitionSamplingFactory()
      const {
        if ((globalPruningConfigPtr_.get() && globalPruningConfigPtr_->shouldUseHoldoutSet())
            || marginalProbabilityCalibratorConfigPtr_->shouldUseHoldoutSet()
            || jointProbabilityCalibratorConfigPtr_->shouldUseHoldoutSet()) {
            return LabelWiseStratifiedBiPartitionSamplingConfig().createPartitionSamplingFactory();
        }

        return NoPartitionSamplingConfig().createPartitionSamplingFactory();
    }

}
