#include "boosting/prediction/predictor_probability_marginalized.hpp"

#include "boosting/prediction/predictor_probability_common.hpp"
#include "boosting/prediction/transformation_probability_marginalized.hpp"
#include "common/prediction/probability_calibration_no.hpp"

#include <stdexcept>

namespace boosting {

    template<typename FeatureMatrix, typename Model>
    static inline std::unique_ptr<IProbabilityPredictor> createPredictor(
      const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels, uint32 numThreads,
      const LabelVectorSet* labelVectorSet,
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
      const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
      const IJointProbabilityFunctionFactory& jointProbabilityFunctionFactory) {
        if (!labelVectorSet) {
            throw std::runtime_error(
              "Information about the label vectors that have been encountered in the training data is required for "
              "predicting binary labels, but no such information is provided by the model. Most probably, the model "
              "was intended to use a different prediction method when it has been trained.");
        }

        std::unique_ptr<IProbabilityTransformation> probabilityTransformationPtr;

        if (labelVectorSet->getNumLabelVectors() > 0) {
            probabilityTransformationPtr = std::make_unique<MarginalizedProbabilityTransformation>(
              *labelVectorSet, jointProbabilityFunctionFactory.create(marginalProbabilityCalibrationModel,
                                                                      jointProbabilityCalibrationModel));
        }

        return std::make_unique<ProbabilityPredictor<FeatureMatrix, Model>>(featureMatrix, model, numLabels, numThreads,
                                                                            std::move(probabilityTransformationPtr));
    }

    /**
     * Allows to create instances of the type `IProbabilityPredictor` that allow to predict label-wise probabilities for
     * given query examples by marginalizing over the joint probabilities of known label vectors.
     */
    class MarginalizedProbabilityPredictorFactory final : public IProbabilityPredictorFactory {
        private:

            const std::unique_ptr<IJointProbabilityFunctionFactory> jointProbabilityFunctionFactoryPtr_;

            const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel_;

            const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel_;

            const uint32 numThreads_;

        public:

            /**
             * @param jointProbabilityFunctionFactoryPtr    An unique pointer to an object of type
             *                                              `IJointProbabilityFunctionFactory` that allows to create
             *                                              implementations of the function to be used to transform
             *                                              regression scores that are predicted for an example into
             *                                              joint probabilities
             * @param marginalProbabilityCalibrationModel   A pointer to an object of type
             *                                              `IMarginalProbabilityCalibrationModel` to be used for the
             *                                              calibration of marginal probabilities or a null pointer, if
             *                                              no such model is available
             * @param jointProbabilityCalibrationModel      A pointer to an object of type
             *                                              `IJointProbabilityCalibrationModel` to be used for the
             *                                              calibration of joint probabilities or a null pointer, if no
             *                                              such model is available
             * @param numThreads                            The number of CPU threads to be used to make predictions for
             *                                              different query examples in parallel. Must be at least 1
             */
            MarginalizedProbabilityPredictorFactory(
              std::unique_ptr<IJointProbabilityFunctionFactory> jointProbabilityFunctionFactoryPtr,
              const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel, uint32 numThreads)
                : jointProbabilityFunctionFactoryPtr_(std::move(jointProbabilityFunctionFactoryPtr)),
                  marginalProbabilityCalibrationModel_(marginalProbabilityCalibrationModel),
                  jointProbabilityCalibrationModel_(jointProbabilityCalibrationModel), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(
              const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return createPredictor(featureMatrix, model, numLabels, numThreads_, labelVectorSet,
                                       marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                                            : marginalProbabilityCalibrationModel,
                                       jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                                         : jointProbabilityCalibrationModel,
                                       *jointProbabilityFunctionFactoryPtr_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(
              const CsrConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return createPredictor(featureMatrix, model, numLabels, numThreads_, labelVectorSet,
                                       marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                                            : marginalProbabilityCalibrationModel,
                                       jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                                         : jointProbabilityCalibrationModel,
                                       *jointProbabilityFunctionFactoryPtr_);
            }
    };

    MarginalizedProbabilityPredictorConfig::MarginalizedProbabilityPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(std::move(lossConfigPtr)), multiThreadingConfigPtr_(std::move(multiThreadingConfigPtr)) {}

    bool MarginalizedProbabilityPredictorConfig::isProbabilityCalibrationModelUsed() const {
        return noMarginalProbabilityCalibrationModelPtr_ == nullptr;
    }

    IMarginalizedProbabilityPredictorConfig& MarginalizedProbabilityPredictorConfig::setUseProbabilityCalibrationModel(
      bool useProbabilityCalibrationModel) {
        noMarginalProbabilityCalibrationModelPtr_ =
          useProbabilityCalibrationModel ? nullptr : createNoProbabilityCalibrationModel();
        noJointProbabilityCalibrationModelPtr_ =
          useProbabilityCalibrationModel ? nullptr : createNoProbabilityCalibrationModel();
        return *this;
    }

    std::unique_ptr<IProbabilityPredictorFactory> MarginalizedProbabilityPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IJointProbabilityFunctionFactory> jointProbabilityFunctionFactoryPtr =
          lossConfigPtr_->createJointProbabilityFunctionFactory();

        if (jointProbabilityFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<MarginalizedProbabilityPredictorFactory>(
              std::move(jointProbabilityFunctionFactoryPtr), noMarginalProbabilityCalibrationModelPtr_.get(),
              noJointProbabilityCalibrationModelPtr_.get(), numThreads);
        } else {
            return nullptr;
        }
    }

    bool MarginalizedProbabilityPredictorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

}
