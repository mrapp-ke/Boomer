#include "boosting/prediction/predictor_binary_gfm.hpp"

#include "boosting/prediction/predictor_binary_common.hpp"
#include "boosting/prediction/transformation_binary_gfm.hpp"
#include "common/prediction/probability_calibration_no.hpp"

#include <stdexcept>

namespace boosting {

    static inline std::unique_ptr<IBinaryTransformation> createBinaryTransformation(
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

        std::unique_ptr<IBinaryTransformation> binaryTransformationPtr;

        if (labelVectorSet->getNumLabelVectors() > 0) {
            std::unique_ptr<IJointProbabilityFunction> jointProbabilityFunctionPtr =
              jointProbabilityFunctionFactory.create(marginalProbabilityCalibrationModel,
                                                     jointProbabilityCalibrationModel);
            binaryTransformationPtr =
              std::make_unique<GfmBinaryTransformation>(*labelVectorSet, std::move(jointProbabilityFunctionPtr));
        }

        return binaryTransformationPtr;
    }

    template<typename FeatureMatrix, typename Model>
    static inline std::unique_ptr<IBinaryPredictor> createPredictor(
      const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels, uint32 numThreads,
      const LabelVectorSet* labelVectorSet,
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
      const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
      const IJointProbabilityFunctionFactory& jointProbabilityFunctionFactory) {
        std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
          createBinaryTransformation(labelVectorSet, marginalProbabilityCalibrationModel,
                                     jointProbabilityCalibrationModel, jointProbabilityFunctionFactory);
        return std::make_unique<BinaryPredictor<FeatureMatrix, Model>>(featureMatrix, model, numLabels, numThreads,
                                                                       std::move(binaryTransformationPtr));
    }

    /**
     * Allows to create instances of the type `IBinaryPredictor` that allow to predict whether individual labels of
     * given query examples are relevant or irrelevant by discretizing the regression scores or probability estimates
     * that are predicted for each label according to the general F-measure maximizer (GFM).
     */
    class GfmBinaryPredictorFactory final : public IBinaryPredictorFactory {
        private:

            const std::unique_ptr<IJointProbabilityFunctionFactory> jointProbabilityFunctionFactoryPtr_;

            const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel_;

            const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel_;

            const uint32 numThreads_;

        public:

            /**
             * @param jointProbabilityFunctionFactoryPtr    An unique pointer to an object of type
             *                                              `IJointProbabilityFunctionFactory` that allows to create
             *                                              implementations of the transformation function to be used to
             *                                              transform regression scores that are predicted for an
             *                                              example into a joint probability
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
            GfmBinaryPredictorFactory(
              std::unique_ptr<IJointProbabilityFunctionFactory> jointProbabilityFunctionFactoryPtr,
              const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel, uint32 numThreads)
                : jointProbabilityFunctionFactoryPtr_(std::move(jointProbabilityFunctionFactoryPtr)),
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
                                       marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                                            : marginalProbabilityCalibrationModel,
                                       jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                                         : jointProbabilityCalibrationModel,
                                       *jointProbabilityFunctionFactoryPtr_);
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
                                       marginalProbabilityCalibrationModel_ ? *marginalProbabilityCalibrationModel_
                                                                            : marginalProbabilityCalibrationModel,
                                       jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                                         : jointProbabilityCalibrationModel,
                                       *jointProbabilityFunctionFactoryPtr_);
            }
    };

    template<typename FeatureMatrix, typename Model>
    static inline std::unique_ptr<ISparseBinaryPredictor> createSparsePredictor(
      const FeatureMatrix& featureMatrix, const Model& model, uint32 numLabels, uint32 numThreads,
      const LabelVectorSet* labelVectorSet,
      const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
      const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
      const IJointProbabilityFunctionFactory& jointProbabilityFunctionFactory) {
        std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
          createBinaryTransformation(labelVectorSet, marginalProbabilityCalibrationModel,
                                     jointProbabilityCalibrationModel, jointProbabilityFunctionFactory);
        return std::make_unique<SparseBinaryPredictor<FeatureMatrix, Model>>(
          featureMatrix, model, numLabels, numThreads, std::move(binaryTransformationPtr));
    }

    /**
     * Allows to create instances of the type `ISparseBinaryPredictor` that allow to predict whether individual labels
     * of given query examples are relevant or irrelevant by discretizing the regression scores or probability estimates
     * that are predicted for each label according to the general F-measure maximizer (GFM).
     */
    class GfmSparseBinaryPredictorFactory final : public ISparseBinaryPredictorFactory {
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
             *                                              a joint probability
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
            GfmSparseBinaryPredictorFactory(
              std::unique_ptr<IJointProbabilityFunctionFactory> jointProbabilityFunctionFactoryPtr,
              const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel* jointProbabilityCalibrationModel, uint32 numThreads)
                : jointProbabilityFunctionFactoryPtr_(std::move(jointProbabilityFunctionFactoryPtr)),
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
                return createSparsePredictor(featureMatrix, model, numLabels, numThreads_, labelVectorSet,
                                             marginalProbabilityCalibrationModel_
                                               ? *marginalProbabilityCalibrationModel_
                                               : marginalProbabilityCalibrationModel,
                                             jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                                               : jointProbabilityCalibrationModel,
                                             *jointProbabilityFunctionFactoryPtr_);
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
                return createSparsePredictor(featureMatrix, model, numLabels, numThreads_, labelVectorSet,
                                             marginalProbabilityCalibrationModel_
                                               ? *marginalProbabilityCalibrationModel_
                                               : marginalProbabilityCalibrationModel,
                                             jointProbabilityCalibrationModel_ ? *jointProbabilityCalibrationModel_
                                                                               : jointProbabilityCalibrationModel,
                                             *jointProbabilityFunctionFactoryPtr_);
            }
    };

    GfmBinaryPredictorConfig::GfmBinaryPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(std::move(lossConfigPtr)), multiThreadingConfigPtr_(std::move(multiThreadingConfigPtr)) {}

    bool GfmBinaryPredictorConfig::isProbabilityCalibrationModelUsed() const {
        return noMarginalProbabilityCalibrationModelPtr_ == nullptr;
    }

    IGfmBinaryPredictorConfig& GfmBinaryPredictorConfig::setUseProbabilityCalibrationModel(
      bool useProbabilityCalibrationModel) {
        noMarginalProbabilityCalibrationModelPtr_ =
          useProbabilityCalibrationModel ? nullptr : createNoProbabilityCalibrationModel();
        noJointProbabilityCalibrationModelPtr_ =
          useProbabilityCalibrationModel ? nullptr : createNoProbabilityCalibrationModel();
        return *this;
    }

    std::unique_ptr<IBinaryPredictorFactory> GfmBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IJointProbabilityFunctionFactory> jointProbabilityFunctionFactoryPtr =
          lossConfigPtr_->createJointProbabilityFunctionFactory();

        if (jointProbabilityFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<GfmBinaryPredictorFactory>(
              std::move(jointProbabilityFunctionFactoryPtr), noMarginalProbabilityCalibrationModelPtr_.get(),
              noJointProbabilityCalibrationModelPtr_.get(), numThreads);
        } else {
            return nullptr;
        }
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> GfmBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IJointProbabilityFunctionFactory> jointProbabilityFunctionFactoryPtr =
          lossConfigPtr_->createJointProbabilityFunctionFactory();

        if (jointProbabilityFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<GfmSparseBinaryPredictorFactory>(
              std::move(jointProbabilityFunctionFactoryPtr), noMarginalProbabilityCalibrationModelPtr_.get(),
              noJointProbabilityCalibrationModelPtr_.get(), numThreads);
        } else {
            return nullptr;
        }
    }

    bool GfmBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

}
