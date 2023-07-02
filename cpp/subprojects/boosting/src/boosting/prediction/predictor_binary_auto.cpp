#include "boosting/prediction/predictor_binary_auto.hpp"

#include "boosting/prediction/predictor_binary_example_wise.hpp"
#include "boosting/prediction/predictor_binary_label_wise.hpp"

namespace boosting {

    AutomaticBinaryPredictorConfig::AutomaticBinaryPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    std::unique_ptr<IBinaryPredictorFactory> AutomaticBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        if (lossConfigPtr_->isDecomposable()) {
            return LabelWiseBinaryPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
              .createPredictorFactory(featureMatrix, numLabels);
        } else {
            return ExampleWiseBinaryPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
              .createPredictorFactory(featureMatrix, numLabels);
        }
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> AutomaticBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        if (lossConfigPtr_->isDecomposable()) {
            return LabelWiseBinaryPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
              .createSparsePredictorFactory(featureMatrix, numLabels);
        } else {
            return ExampleWiseBinaryPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_)
              .createSparsePredictorFactory(featureMatrix, numLabels);
        }
    }

    bool AutomaticBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        if (lossConfigPtr_->isDecomposable()) {
            return LabelWiseBinaryPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_).isLabelVectorSetNeeded();
        } else {
            return ExampleWiseBinaryPredictorConfig(lossConfigPtr_, multiThreadingConfigPtr_).isLabelVectorSetNeeded();
        }
    }

}
