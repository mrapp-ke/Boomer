#include "boosting/output/predictor_classification_auto.hpp"
#include "boosting/output/predictor_classification_example_wise.hpp"
#include "boosting/output/predictor_classification_label_wise.hpp"
#include "boosting/losses/loss_example_wise.hpp"


namespace boosting {

    static inline bool isExampleWisePredictorPreferred(const ILossConfig* lossConfig) {
        return dynamic_cast<const IExampleWiseLossConfig*>(lossConfig) != nullptr;
    }

    AutomaticClassificationPredictorConfig::AutomaticClassificationPredictorConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr,
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {

    }

    std::unique_ptr<IClassificationPredictorFactory> AutomaticClassificationPredictorConfig::createClassificationPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        if (isExampleWisePredictorPreferred(lossConfigPtr_.get())) {
            return ExampleWiseClassificationPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .createClassificationPredictorFactory(featureMatrix, numLabels);
        } else {
            return LabelWiseClassificationPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .createClassificationPredictorFactory(featureMatrix, numLabels);
        }
    }

    std::unique_ptr<ILabelSpaceInfo> AutomaticClassificationPredictorConfig::createLabelSpaceInfo(
            const IRowWiseLabelMatrix& labelMatrix) const {
        if (isExampleWisePredictorPreferred(lossConfigPtr_.get())) {
            return ExampleWiseClassificationPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .createLabelSpaceInfo(labelMatrix);
        } else {
            return LabelWiseClassificationPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
                .createLabelSpaceInfo(labelMatrix);
        }
    }

}
