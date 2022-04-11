#include "boosting/output/predictor_probability_label_wise.hpp"
#include "boosting/output/probability_function.hpp"
#include "predictor_common.hpp"
#include "omp.h"


namespace boosting {

    static inline void applyTransformationFunction(CContiguousConstView<float64>::value_const_iterator originalIterator,
                                                   CContiguousView<float64>::value_iterator transformedIterator,
                                                   uint32 numElements,
                                                   const IProbabilityFunction& probabilityFunction) {
        for (uint32 i = 0; i < numElements; i++) {
            float64 originalValue = originalIterator[i];
            float64 transformedValue = probabilityFunction.transform(originalValue);
            transformedIterator[i] = transformedValue;
        }
    }

    /**
     * An implementation of the type `ILabelWiseProbabilityPredictor` that allows to predict label-wise probabilities
     * for given query examples, which estimate the chance of individual labels to be relevant, by summing up the scores
     * that are provided by individual rules of an existing rule-based models and transforming the aggregated scores
     * into probabilities in [0, 1] according to a certain transformation function that is applied to each label
     * individually.
     *
     * @tparam Model The type of the rule-based model that is used to obtain predictions
     */
    template<typename Model>
    class LabelWiseProbabilityPredictor final : public IProbabilityPredictor {

        private:

            const Model& model_;

            std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param model                     A reference to an object of template type `Model` that should be used to
             *                                  obtain predictions
             * @param probabilityFunctionPtr    An unique pointer to an object of type `IProbabilityFunction` that
             *                                  should be used to transform predicted scores into probabilities
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             */
            LabelWiseProbabilityPredictor(const Model& model,
                                          std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr,
                                          uint32 numThreads)
                : model_(model), probabilityFunctionPtr_(std::move(probabilityFunctionPtr)), numThreads_(numThreads) {

            }

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<float64>> predict(
                    const CContiguousConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
                    std::make_unique<DensePredictionMatrix<float64>>(numExamples, numLabels);
                const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
                CContiguousView<float64>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                const Model* modelPtr = &model_;
                const IProbabilityFunction* probabilityFunctionPtr = probabilityFunctionPtr_.get();

                #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
                firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) \
                firstprivate(probabilityFunctionPtr) schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};

                    for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                        const RuleList::Rule& rule = *it;
                        applyRule(rule, featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                                  &scoreVector[0]);
                    }

                    applyTransformationFunction(&scoreVector[0], predictionMatrixRawPtr->row_values_begin(i), numLabels,
                                                *probabilityFunctionPtr);
                    delete[] scoreVector;
                }

                return predictionMatrixPtr;
            }

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<float64>> predict(const CsrConstView<const float32>& featureMatrix,
                                                                    uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                std::unique_ptr<DensePredictionMatrix<float64>> predictionMatrixPtr =
                    std::make_unique<DensePredictionMatrix<float64>>(numExamples, numLabels);
                const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
                CContiguousView<float64>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                const Model* modelPtr = &model_;
                const IProbabilityFunction* probabilityFunctionPtr = probabilityFunctionPtr_.get();

                #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(numFeatures) \
                firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) \
                firstprivate(probabilityFunctionPtr) schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    float32* tmpArray1 = new float32[numFeatures];
                    uint32* tmpArray2 = new uint32[numFeatures] {};
                    uint32 n = 1;

                    for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                        const RuleList::Rule& rule = *it;
                        applyRuleCsr(rule, featureMatrixPtr->row_indices_cbegin(i),
                                     featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                                     featureMatrixPtr->row_values_cend(i), &scoreVector[0], &tmpArray1[0],
                                     &tmpArray2[0], n);
                        n++;
                    }

                    applyTransformationFunction(&scoreVector[0], predictionMatrixRawPtr->row_values_begin(i), numLabels,
                                                *probabilityFunctionPtr);
                    delete[] scoreVector;
                    delete[] tmpArray1;
                    delete[] tmpArray2;
                }

                return predictionMatrixPtr;
            }

    };

    /**
     * Allows to create instances of the type `IProbabilityPredictor` that allow to predict label-wise probabilities
     * for given query examples, which estimate the chance of individual labels to be relevant, by summing up the scores
     * that are provided by individual rules of an existing rule-based models and transforming the aggregated scores
     * into probabilities in [0, 1] according to a certain transformation function that is applied to each label
     * individually.
     */
    class LabelWiseProbabilityPredictorFactory final : public IProbabilityPredictorFactory {

        private:

            std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param probabilityFunctionFactoryPtr An unique pointer to an object of type `IProbabilityFunctionFactory`
             *                                      that allows to create implementations of the transformation function
             *                                      to be used to transform predicted scores into probabilities
             * @param numThreads                    The number of CPU threads to be used to make predictions for
             *                                      different query examples in parallel. Must be at least 1
             */
            LabelWiseProbabilityPredictorFactory(
                    std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr, uint32 numThreads)
                : probabilityFunctionFactoryPtr_(std::move(probabilityFunctionFactoryPtr)), numThreads_(numThreads) {

            }

            /**
             * @see `IProbabilityPredictorFactory::create`
             */
            std::unique_ptr<IProbabilityPredictor> create(const RuleList& model,
                                                          const LabelVectorSet* labelVectorSet) const override {
                std::unique_ptr<IProbabilityFunction> probabilityFunctionPtr = probabilityFunctionFactoryPtr_->create();
                return std::make_unique<LabelWiseProbabilityPredictor<RuleList>>(model,
                                                                                 std::move(probabilityFunctionPtr),
                                                                                 numThreads_);
            }

    };

    LabelWiseProbabilityPredictorConfig::LabelWiseProbabilityPredictorConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr,
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {

    }

    std::unique_ptr<IProbabilityPredictorFactory> LabelWiseProbabilityPredictorConfig::createProbabilityPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr =
            lossConfigPtr_->createProbabilityFunctionFactory();

        if (probabilityFunctionFactoryPtr) {
            uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
            return std::make_unique<LabelWiseProbabilityPredictorFactory>(
                std::move(probabilityFunctionFactoryPtr), numThreads);
        } else {
            return nullptr;
        }
    }

}
