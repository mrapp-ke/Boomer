#include "boosting/prediction/predictor_binary_label_wise.hpp"

#include "boosting/prediction/discretization_function_probability.hpp"
#include "boosting/prediction/discretization_function_score.hpp"
#include "boosting/prediction/predictor_binary_common.hpp"
#include "boosting/prediction/transformation_binary_label_wise.hpp"
#include "common/prediction/probability_calibration_no.hpp"

namespace boosting {

    /**
     * Allows to create instances of the type `IBinaryPredictor` that allow to predict whether individual labels of
     * given query examples are relevant or irrelevant by discretizing the regression scores or probability estimates
     * that are predicted for each label individually.
     */
    class LabelWiseBinaryPredictorFactory final : public IBinaryPredictorFactory {
        private:

            const std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr_;

            const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel_;

            const uint32 numThreads_;

        public:

            /**
             * @param discretizationFunctionFactoryPtr      An unique pointer to an object of type
             *                                              `IDiscretizationFunctionFactory` that allows to create the
             *                                              implementation to be used for discretization
             * @param marginalProbabilityCalibrationModel   A pointer to an object of type
             *                                              `IMarginalProbabilityCalibrationModel` to be used for the
             *                                              calibration of marginal probabilities or a null pointer, if
             *                                              no such model is available
             * @param numThreads                            The number of CPU threads to be used to make predictions for
             *                                              different query examples in parallel. Must be at least 1
             */
            LabelWiseBinaryPredictorFactory(
              std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr,
              const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel, uint32 numThreads)
                : discretizationFunctionFactoryPtr_(std::move(discretizationFunctionFactoryPtr)),
                  marginalProbabilityCalibrationModel_(marginalProbabilityCalibrationModel), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IBinaryPredictor> create(
              const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                std::unique_ptr<IDiscretizationFunction> discretizationFunctionPtr =
                  discretizationFunctionFactoryPtr_->create(marginalProbabilityCalibrationModel_
                                                              ? *marginalProbabilityCalibrationModel_
                                                              : marginalProbabilityCalibrationModel);
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<LabelWiseBinaryTransformation>(std::move(discretizationFunctionPtr));
                return std::make_unique<BinaryPredictor<CContiguousConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(binaryTransformationPtr));
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
                std::unique_ptr<IDiscretizationFunction> discretizationFunctionPtr =
                  discretizationFunctionFactoryPtr_->create(marginalProbabilityCalibrationModel_
                                                              ? *marginalProbabilityCalibrationModel_
                                                              : marginalProbabilityCalibrationModel);
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<LabelWiseBinaryTransformation>(std::move(discretizationFunctionPtr));
                return std::make_unique<BinaryPredictor<CsrConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(binaryTransformationPtr));
            }
    };

    /**
     * Allows to create instances of the type `ISparseBinaryPredictor` that allow to predict whether individual labels
     * of given query examples are relevant or irrelevant by discretizing the regression scores or probability estimates
     * that are predicted for each label individually.
     */
    class LabelWiseSparseBinaryPredictorFactory final : public ISparseBinaryPredictorFactory {
        private:

            const std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr_;

            const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel_;

            const uint32 numThreads_;

        public:

            /**
             * @param discretizationFunctionFactoryPtr      An unique pointer to an object of type
             *                                              `IDiscretizationFunctionFactory` that allows to create the
             *                                              implementation to be used for discretization
             * @param marginalProbabilityCalibrationModel   A pointer to an object of type
             *                                              `IMarginalProbabilityCalibrationModel` to be used for the
             *                                              calibration of marginal probabilities or a null pointer, if
             *                                              no such model is available
             * @param numThreads                            The number of CPU threads to be used to make predictions for
             *                                              different query examples in parallel. Must be at least 1
             */
            LabelWiseSparseBinaryPredictorFactory(
              std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr,
              const IMarginalProbabilityCalibrationModel* marginalProbabilityCalibrationModel, uint32 numThreads)
                : discretizationFunctionFactoryPtr_(std::move(discretizationFunctionFactoryPtr)),
                  marginalProbabilityCalibrationModel_(marginalProbabilityCalibrationModel), numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<ISparseBinaryPredictor> create(
              const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
              const LabelVectorSet* labelVectorSet,
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel,
              uint32 numLabels) const override {
                std::unique_ptr<IDiscretizationFunction> discretizationFunctionPtr =
                  discretizationFunctionFactoryPtr_->create(marginalProbabilityCalibrationModel_
                                                              ? *marginalProbabilityCalibrationModel_
                                                              : marginalProbabilityCalibrationModel);
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<LabelWiseBinaryTransformation>(std::move(discretizationFunctionPtr));
                return std::make_unique<SparseBinaryPredictor<CContiguousConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(binaryTransformationPtr));
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
                std::unique_ptr<IDiscretizationFunction> discretizationFunctionPtr =
                  discretizationFunctionFactoryPtr_->create(marginalProbabilityCalibrationModel_
                                                              ? *marginalProbabilityCalibrationModel_
                                                              : marginalProbabilityCalibrationModel);
                std::unique_ptr<IBinaryTransformation> binaryTransformationPtr =
                  std::make_unique<LabelWiseBinaryTransformation>(std::move(discretizationFunctionPtr));
                return std::make_unique<SparseBinaryPredictor<CsrConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_, std::move(binaryTransformationPtr));
            }
    };

    static inline std::unique_ptr<IDiscretizationFunctionFactory> createDiscretizationFunctionFactory(
      bool basedOnProbabilities, const ILossConfig& lossConfig) {
        if (basedOnProbabilities) {
            std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactory =
              lossConfig.createMarginalProbabilityFunctionFactory();

            if (marginalProbabilityFunctionFactory) {
                return std::make_unique<ProbabilityDiscretizationFunctionFactory>(
                  std::move(marginalProbabilityFunctionFactory));
            } else {
                return nullptr;
            }
        } else {
            float64 threshold = lossConfig.getDefaultPrediction();
            return std::make_unique<ScoreDiscretizationFunctionFactory>(threshold);
        }
    }

    LabelWiseBinaryPredictorConfig::LabelWiseBinaryPredictorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : basedOnProbabilities_(false), lossConfigPtr_(lossConfigPtr),
          multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    bool LabelWiseBinaryPredictorConfig::isBasedOnProbabilities() const {
        return basedOnProbabilities_;
    }

    ILabelWiseBinaryPredictorConfig& LabelWiseBinaryPredictorConfig::setBasedOnProbabilities(
      bool basedOnProbabilities) {
        basedOnProbabilities_ = basedOnProbabilities;
        return *this;
    }

    bool LabelWiseBinaryPredictorConfig::isProbabilityCalibrationModelUsed() const {
        return noMarginalProbabilityCalibrationModelPtr_ == nullptr;
    }

    ILabelWiseBinaryPredictorConfig& LabelWiseBinaryPredictorConfig::setUseProbabilityCalibrationModel(
      bool useProbabilityCalibrationModel) {
        noMarginalProbabilityCalibrationModelPtr_ =
          useProbabilityCalibrationModel ? nullptr : createNoProbabilityCalibrationModel();
        return *this;
    }

    std::unique_ptr<IBinaryPredictorFactory> LabelWiseBinaryPredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr =
          createDiscretizationFunctionFactory(basedOnProbabilities_, *lossConfigPtr_);

        if (discretizationFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<LabelWiseBinaryPredictorFactory>(
              std::move(discretizationFunctionFactoryPtr), noMarginalProbabilityCalibrationModelPtr_.get(), numThreads);
        }

        return nullptr;
    }

    std::unique_ptr<ISparseBinaryPredictorFactory> LabelWiseBinaryPredictorConfig::createSparsePredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IDiscretizationFunctionFactory> discretizationFunctionFactoryPtr =
          createDiscretizationFunctionFactory(basedOnProbabilities_, *lossConfigPtr_);

        if (discretizationFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<LabelWiseSparseBinaryPredictorFactory>(
              std::move(discretizationFunctionFactoryPtr), noMarginalProbabilityCalibrationModelPtr_.get(), numThreads);
        }

        return nullptr;
    }

    bool LabelWiseBinaryPredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }
}
