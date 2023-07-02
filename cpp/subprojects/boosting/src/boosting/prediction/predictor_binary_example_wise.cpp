#include "boosting/prediction/predictor_binary_example_wise.hpp"

#include "boosting/prediction/predictor_binary_common.hpp"
#include "boosting/prediction/transformation_binary_example_wise.hpp"
#include "common/prediction/probability_calibration_no.hpp"

#include <stdexcept>

namespace boosting {

    static inline std::unique_ptr<IBinaryTransformation> createBinaryTransformation(
      const LabelVectorSet* labelVectorSet, const IDistanceMeasureFactory& distanceMeasureFactory,
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
      const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) {
        if (!labelVectorSet) {
            throw std::runtime_error(
              "Information about the label vectors that have been encountered in the training data is required for "
              "predicting binary labels, but no such information is provided by the model. Most probably, the model "
              "was intended to use a different prediction method when it has been trained.");
        }

        std::unique_ptr<IBinaryTransformation> binaryTransformationPtr;

        if (labelVectorSet->getNumLabelVectors() > 0) {
            binaryTransformationPtr = std::make_unique<ExampleWiseBinaryTransformation>(
              *labelVectorSet, distanceMeasureFactory.createDistanceMeasure(marginalProbabilityCalibrationModel,
                                                                            jointProbabilityCalibrationModel));
        }

        return binaryTransformationPtr;
    }

    template<typename FeatureMatrix, typename Model>
    static inline std::unique_ptr<IBinaryPredictor> createPredictor(
      const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels, uint32 numThreads,
      const LabelVectorSet* labelVectorSet, const IDistanceMeasureFactory& distanceMeasureFactory,
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
      const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) {
        std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
          createBinaryTransformation(labelVectorSet, distanceMeasureFactory, marginalProbabilityCalibrationModel,
                                     jointProbabilityCalibrationModel);
        return std::make_unique<BinaryPredictor<FeatureMatrix, Model>>(featureMatrix, model, numLabels, numThreads,
                                                                       std::move(binaryTransformationPtr));
    }

    /**
     * Allows to create instances of the type `IBinaryPredictor` that allow to predict known label vectors for given
     * query examples by comparing the predicted regression scores or probability estimates to the label vectors
     * encountered in the training data.
     */
    class ExampleWiseBinaryPredictorFactory final : public IBinaryPredictorFactory {
        private:

            const std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr_;

            const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel_;

            const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel_;

            const uint32 numThreads_;

        public:

            /**
             * @param distanceMeasureFactoryPtr             An unique pointer to an object of type
             *                                              `IDistanceMeasureFactory` that allows to create
             *                                              implementations of the distance measure that should be used
             *                                              to calculate the distance between predicted scores and known
             *                                              label vectors
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
            ExampleWiseBinaryPredictorFactory(
              std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr,
              const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel, uint32 numThreads)
                : distanceMeasureFactoryPtr_(std::move(distanceMeasureFactoryPtr)),
                  marginalProbabilityCalibrationModel_(marginalProbabilityCalibrationModel),
                  jointProbabilityCalibrationModel_(jointProbabilityCalibrationModel), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(
              const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return createPredictor(featureMatrix, model, numLabels, numThreads_, labelVectorSet,
                                       *distanceMeasureFactoryPtr_,
                                       marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                                            : marginalProbabilityCalibrationModel,
                                       jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                                         : jointProbabilityCalibrationModel);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(
              const CsrConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return createPredictor(featureMatrix, model, numLabels, numThreads_, labelVectorSet,
                                       *distanceMeasureFactoryPtr_,
                                       marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                                            : marginalProbabilityCalibrationModel,
                                       jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                                         : jointProbabilityCalibrationModel);
            }
    };

    template<typename FeatureMatrix, typename Model>
    static inline std::unique_ptr<ISparseBinaryPredictor> createSparsePredictor(
      const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels, uint32 numThreads,
      const LabelVectorSet* labelVectorSet, const IDistanceMeasureFactory& distanceMeasureFactory,
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
      const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) {
        std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
          createBinaryTransformation(labelVectorSet, distanceMeasureFactory, marginalProbabilityCalibrationModel,
                                     jointProbabilityCalibrationModel);
        return std::make_unique<SparseBinaryPredictor<FeatureMatrix, Model>>(
          featureMatrix, model, numLabels, numThreads, std::move(binaryTransformationPtr));
    }

    /**
     * Allows to create instances of the type `ISparseBinaryPredictor` that allow to predict known label vectors for
     * given query examples by comparing the predicted regression scores or probability estimates to the label vectors
     * encountered in the training data.
     */
    class ExampleWiseSparseBinaryPredictorFactory final : public ISparseBinaryPredictorFactory {
        private:

            const std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr_;

            const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel_;

            const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel_;

            const uint32 numThreads_;

        public:

            /**
             * @param distanceMeasureFactoryPtr             An unique pointer to an object of type
             *                                              `IDistanceMeasureFactory` that allows to create
             *                                              implementations of the distance measure that should be used
             *                                              to calculate the distance between predicted scores and known
             *                                              label vectors
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
            ExampleWiseSparseBinaryPredictorFactory(
              std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr,
              const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel, uint32 numThreads)
                : distanceMeasureFactoryPtr_(std::move(distanceMeasureFactoryPtr)),
                  marginalProbabilityCalibrationModel_(marginalProbabilityCalibrationModel),
                  jointProbabilityCalibrationModel_(jointProbabilityCalibrationModel), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(
              const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return createSparsePredictor(
                  featureMatrix, model, numLabels, numThreads_, labelVectorSet, *distanceMeasureFactoryPtr_,
                  marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                       : marginalProbabilityCalibrationModel,
                  jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                    : jointProbabilityCalibrationModel);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(
              const CsrConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                return createSparsePredictor(
                  featureMatrix, model, numLabels, numThreads_, labelVectorSet, *distanceMeasureFactoryPtr_,
                  marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                       : marginalProbabilityCalibrationModel,
                  jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                    : jointProbabilityCalibrationModel);
            }
    };

    static inline std::unique_ptr<IDistanceMeasureFactory> createDistanceMeasureFactory(bool basedOnProbabilities,
                                                                                        const ILossConfig& lossConfig) {
        if (basedOnProbabilities) {
            return lossConfig.createJointProbabilityFunctionFactory();
        } else {
            return lossConfig.createDistanceMeasureFactory();
        }
    }

    ExampleWiseBinaryPredictorConfig::ExampleWiseBinaryPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : basedOnProbabilities_(false), lossConfigPtr_(lossConfigPtr),
          multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    bool ExampleWiseBinaryPredictorConfig::isBasedOnProbabilities() const {
        return basedOnProbabilities_;
    }

    IExampleWiseBinaryPredictorConfig& ExampleWiseBinaryPredictorConfig::setBasedOnProbabilities(
      bool basedOnProbabilities) {
        basedOnProbabilities_ = basedOnProbabilities;
        return *this;
    }

    bool ExampleWiseBinaryPredictorConfig::isProbabilityCalibrationModelUsed() const {
        return noMarginalProbabilityCalibrationModelPtr_ == nullptr;
    }

    IExampleWiseBinaryPredictorConfig& ExampleWiseBinaryPredictorConfig::setUseProbabilityCalibrationModel(
      bool useProbabilityCalibrationModel) {
        noMarginalProbabilityCalibrationModelPtr_ =
          useProbabilityCalibrationModel ? nullptr : createNoProbabilityCalibrationModel();
        noJointProbabilityCalibrationModelPtr_ =
          useProbabilityCalibrationModel ? nullptr : createNoProbabilityCalibrationModel();
        return *this;
    }

    std::unique_ptr<IBinaryPredictorFactory> ExampleWiseBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr =
          createDistanceMeasureFactory(basedOnProbabilities_, *lossConfigPtr_);

        if (distanceMeasureFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<ExampleWiseBinaryPredictorFactory>(
              std::move(distanceMeasureFactoryPtr), noMarginalProbabilityCalibrationModelPtr_.get(),
              noJointProbabilityCalibrationModelPtr_.get(), numThreads);
        }

        return nullptr;
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> ExampleWiseBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IDistanceMeasureFactory> distanceMeasureFactoryPtr =
          createDistanceMeasureFactory(basedOnProbabilities_, *lossConfigPtr_);

        if (distanceMeasureFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<ExampleWiseSparseBinaryPredictorFactory>(
              std::move(distanceMeasureFactoryPtr), noMarginalProbabilityCalibrationModelPtr_.get(),
              noJointProbabilityCalibrationModelPtr_.get(), numThreads);
        }

        return nullptr;
    }

    bool ExampleWiseBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

}
