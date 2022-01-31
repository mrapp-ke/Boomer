#include "boosting/multi_threading/parallel_rule_refinement_auto.hpp"
#include "boosting/losses/loss_example_wise.hpp"
#include "boosting/rule_evaluation/head_type_single.hpp"
#include "common/sampling/feature_sampling_no.hpp"
#include "common/util/threads.hpp"


namespace boosting {

    AutoParallelRuleRefinementConfig::AutoParallelRuleRefinementConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr, const std::unique_ptr<IHeadConfig>& headConfigPtr,
            const std::unique_ptr<IFeatureSamplingConfig>& featureSamplingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), headConfigPtr_(headConfigPtr),
          featureSamplingConfigPtr_(featureSamplingConfigPtr) {

    }

    uint32 AutoParallelRuleRefinementConfig::getNumThreads(const IFeatureMatrix& featureMatrix,
                                                           uint32 numLabels) const {
        if (dynamic_cast<const IExampleWiseLossConfig*>(lossConfigPtr_.get())
                && !dynamic_cast<const SingleLabelHeadConfig*>(headConfigPtr_.get())) {
            return 1;
        } else {
            if (featureMatrix.isSparse()
                    && !dynamic_cast<const NoFeatureSamplingConfig*>(featureSamplingConfigPtr_.get())) {
                return 1;
            } else {
                return getNumAvailableThreads(0);
            }
        }
    };

}
