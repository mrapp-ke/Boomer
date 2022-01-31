#include "boosting/output/predictor_classification_example_wise.hpp"
#include "common/data/arrays.hpp"
#include "predictor_common.hpp"
#include "omp.h"
#include <algorithm>


namespace boosting {

    static inline const LabelVector* findClosestLabelVector(const float64* scoresBegin, const float64* scoresEnd,
                                                            const ISimilarityMeasure& measure,
                                                            const LabelVectorSet* labelVectorSet) {
        const LabelVector* closestLabelVector = nullptr;

        if (labelVectorSet) {
            float64 bestScore = 0;
            uint32 bestCount = 0;

            for (auto it = labelVectorSet->cbegin(); it != labelVectorSet->cend(); it++) {
                const auto& entry = *it;
                const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
                uint32 count = entry.second;
                float64 score = measure.measureSimilarity(*labelVectorPtr, scoresBegin, scoresEnd);

                if (!closestLabelVector || score < bestScore || (score == bestScore && count > bestCount)) {
                    closestLabelVector = labelVectorPtr.get();
                    bestScore = score;
                    bestCount = count;
                }
            }
        }

        return closestLabelVector;
    }

    static inline void predictLabelVector(CContiguousView<uint8>::value_iterator predictionIterator, uint32 numElements,
                                          const LabelVector* labelVector) {
        setArrayToZeros(predictionIterator, numElements);

        if (labelVector) {
            uint32 numIndices = labelVector->getNumElements();
            LabelVector::const_iterator indexIterator = labelVector->cbegin();

            for (uint32 i = 0; i < numIndices; i++) {
                uint32 labelIndex = indexIterator[i];
                predictionIterator[labelIndex] = 1;
            }
        }
    }

    static inline uint32 predictLabelVector(BinaryLilMatrix::Row& row, const LabelVector* labelVector) {
        uint32 numNonZeroElements = 0;

        if (labelVector) {
            uint32 numIndices = labelVector->getNumElements();
            LabelVector::const_iterator indexIterator = labelVector->cbegin();

            if (numIndices > 0) {
                uint32 labelIndex = indexIterator[0];
                row.emplace_front(labelIndex);
                numNonZeroElements++;
                BinaryLilMatrix::Row::iterator it = row.begin();

                for (uint32 i = 1; i < numIndices; i++) {
                    labelIndex = indexIterator[i];
                    it = row.emplace_after(it, labelIndex);
                    numNonZeroElements++;
                }
            }
        }

        return numNonZeroElements;
    }

    /**
     * An implementation of the type `IExampleWiseClassificationPredictor` that allows to predict known label vectors
     * for given query examples by summing up the scores that are provided by an existing rule-based model and comparing
     * the aggregated score vector to the known label vectors according to a certain distance measure. The label vector
     * that is closest to the aggregated score vector is finally predicted.
     *
     * @tparam Model The type of the rule-based model that is used to obtain predictions
     */
    template<typename Model>
    class ExampleWiseClassificationPredictor final : public IClassificationPredictor {

        private:

            const Model& model_;

            const LabelVectorSet* labelVectorSet_;

            std::unique_ptr<ISimilarityMeasure> similarityMeasurePtr_;

            uint32 numThreads_;

        public:

            /**
             * @param model                 A reference to an object of template type `Model` that should be used to
             *                              obtain predictions
             * @param labelVectorSet        A pointer to an object of type `LabelVectorSet` that stores all known label
             *                              vectors or a null pointer, if no such set is available
             * @param similarityMeasurePtr  An unique pointer to an object of type `ISimilarityMeasure` that implements
             *                              the similarity measure that should be used to quantify the similarity
             *                              between predictions and known label vectors
             * @param numThreads            The number of CPU threads to be used to make predictions for different query
             *                              examples in parallel. Must be at least 1
             */
            ExampleWiseClassificationPredictor(const Model& model, const LabelVectorSet* labelVectorSet,
                                               std::unique_ptr<ISimilarityMeasure> similarityMeasurePtr,
                                               uint32 numThreads)
                : model_(model), labelVectorSet_(labelVectorSet), similarityMeasurePtr_(std::move(similarityMeasurePtr)),
                  numThreads_(numThreads) {

            }

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<uint8>> predict(
                    const CContiguousConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                    std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels);
                const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
                CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                const Model* modelPtr = &model_;
                const LabelVectorSet* labelVectorSetPtr = labelVectorSet_;
                const ISimilarityMeasure* similarityMeasureRawPtr = similarityMeasurePtr_.get();

                #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
                firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) \
                firstprivate(similarityMeasureRawPtr) firstprivate(labelVectorSetPtr) schedule(dynamic) \
                num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRules(*modelPtr, featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                               &scoreVector[0]);
                    const LabelVector* closestLabelVector = findClosestLabelVector(&scoreVector[0],
                                                                                   &scoreVector[numLabels],
                                                                                   *similarityMeasureRawPtr,
                                                                                   labelVectorSetPtr);
                    predictLabelVector(predictionMatrixRawPtr->row_values_begin(i), numLabels, closestLabelVector);
                    delete[] scoreVector;
                }

                return predictionMatrixPtr;
            }

            /**
             * @see `IPredictor::predict`
             */
            std::unique_ptr<DensePredictionMatrix<uint8>> predict(const CsrConstView<const float32>& featureMatrix,
                                                                  uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                std::unique_ptr<DensePredictionMatrix<uint8>> predictionMatrixPtr =
                    std::make_unique<DensePredictionMatrix<uint8>>(numExamples, numLabels);
                const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
                CContiguousView<uint8>* predictionMatrixRawPtr = predictionMatrixPtr.get();
                const Model* modelPtr = &model_;
                const LabelVectorSet* labelVectorSetPtr = labelVectorSet_;
                const ISimilarityMeasure* similarityMeasureRawPtr = similarityMeasurePtr_.get();

                #pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) firstprivate(numLabels) \
                firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixRawPtr) \
                firstprivate(similarityMeasureRawPtr) firstprivate(labelVectorSetPtr) schedule(dynamic) \
                num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                                  featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                                  featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                    const LabelVector* closestLabelVector = findClosestLabelVector(&scoreVector[0],
                                                                                   &scoreVector[numLabels],
                                                                                   *similarityMeasureRawPtr,
                                                                                   labelVectorSetPtr);
                    predictLabelVector(predictionMatrixRawPtr->row_values_begin(i), numLabels, closestLabelVector);
                    delete[] scoreVector;
                }

                return predictionMatrixPtr;
            }

            /**
             * @see `ISparsePredictor::predictSparse`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
                    const CContiguousConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                BinaryLilMatrix lilMatrix(numExamples);
                const CContiguousConstView<const float32>* featureMatrixPtr = &featureMatrix;
                BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
                const Model* modelPtr = &model_;
                const LabelVectorSet* labelVectorSetPtr = labelVectorSet_;
                const ISimilarityMeasure* similarityMeasureRawPtr = similarityMeasurePtr_.get();
                uint32 numNonZeroElements = 0;

                #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) \
                firstprivate(numLabels) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
                firstprivate(predictionMatrixPtr) firstprivate(similarityMeasureRawPtr) \
                firstprivate(labelVectorSetPtr) schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRules(*modelPtr, featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                               &scoreVector[0]);
                    const LabelVector* closestLabelVector = findClosestLabelVector(&scoreVector[0],
                                                                                   &scoreVector[numLabels],
                                                                                   *similarityMeasureRawPtr,
                                                                                   labelVectorSetPtr);
                    numNonZeroElements += predictLabelVector(predictionMatrixPtr->getRow(i), closestLabelVector);
                    delete[] scoreVector;
                }

                return createBinarySparsePredictionMatrix(lilMatrix, numLabels, numNonZeroElements);
            }

            /**
             * @see `ISparsePredictor::predictSparse`
             */
            std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
                    const CsrConstView<const float32>& featureMatrix, uint32 numLabels) const override {
                uint32 numExamples = featureMatrix.getNumRows();
                uint32 numFeatures = featureMatrix.getNumCols();
                BinaryLilMatrix lilMatrix(numExamples);
                const CsrConstView<const float32>* featureMatrixPtr = &featureMatrix;
                BinaryLilMatrix* predictionMatrixPtr = &lilMatrix;
                const Model* modelPtr = &model_;
                const LabelVectorSet* labelVectorSetPtr = labelVectorSet_;
                const ISimilarityMeasure* similarityMeasureRawPtr = similarityMeasurePtr_.get();
                uint32 numNonZeroElements = 0;

                #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) \
                firstprivate(numFeatures) firstprivate(numLabels) firstprivate(modelPtr) \
                firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) firstprivate(similarityMeasureRawPtr) \
                firstprivate(labelVectorSetPtr) schedule(dynamic) num_threads(numThreads_)
                for (int64 i = 0; i < numExamples; i++) {
                    float64* scoreVector = new float64[numLabels] {};
                    applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                                  featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                                  featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
                    const LabelVector* closestLabelVector = findClosestLabelVector(&scoreVector[0],
                                                                                   &scoreVector[numLabels],
                                                                                   *similarityMeasureRawPtr,
                                                                                   labelVectorSetPtr);
                    numNonZeroElements += predictLabelVector(predictionMatrixPtr->getRow(i), closestLabelVector);
                    delete[] scoreVector;
                }

                return createBinarySparsePredictionMatrix(lilMatrix, numLabels, numNonZeroElements);
            }

    };

    /**
     * Allows to create instances of the type `IClassificationPredictor` that allow to predict known label vectors for
     * given query examples by summing up the scores that are provided by an existing rule-based model and comparing the
     * aggregated score vector to the known label vectors according to a certain distance measure. The label vector that
     * is closest to the aggregated score vector is finally predicted.
     */
    class ExampleWiseClassificationPredictorFactory final : public IClassificationPredictorFactory {

        private:

            std::unique_ptr<ISimilarityMeasureFactory> similarityMeasureFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param similarityMeasureFactoryPtr   An unique pointer to an object of type `ISimilarityMeasureFactory`
             *                                      that allows to create implementations of the similarity measure
             *                                      that should be used to quantify the similarity between predictions
             *                                      and known label vectors
             * @param numThreads                    The number of CPU threads to be used to make predictions for
             *                                      different query examples in parallel. Must be at least 1
             */
            ExampleWiseClassificationPredictorFactory(
                    std::unique_ptr<ISimilarityMeasureFactory> similarityMeasureFactoryPtr, uint32 numThreads)
                : similarityMeasureFactoryPtr_(std::move(similarityMeasureFactoryPtr)), numThreads_(numThreads) {

            }

            /**
             * @see `IClassificationPredictorFactory::create`
             */
            std::unique_ptr<IClassificationPredictor> create(const RuleList& model,
                                                             const LabelVectorSet* labelVectorSet) const override {
                std::unique_ptr<ISimilarityMeasure> similarityMeasurePtr =
                    similarityMeasureFactoryPtr_->createSimilarityMeasure();
                return std::make_unique<ExampleWiseClassificationPredictor<RuleList>>(model, labelVectorSet,
                                                                                      std::move(similarityMeasurePtr),
                                                                                      numThreads_);
            }

    };

    ExampleWiseClassificationPredictorConfig::ExampleWiseClassificationPredictorConfig(
            const std::unique_ptr<ILossConfig>& lossConfigPtr,
            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : lossConfigPtr_(lossConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr) {

    }

    std::unique_ptr<IClassificationPredictorFactory> ExampleWiseClassificationPredictorConfig::createClassificationPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const {
        std::unique_ptr<ISimilarityMeasureFactory> similarityMeasureFactoryPtr =
            lossConfigPtr_->createSimilarityMeasureFactory();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<ExampleWiseClassificationPredictorFactory>(
            std::move(similarityMeasureFactoryPtr), numThreads);
    }

    std::unique_ptr<ILabelSpaceInfo> ExampleWiseClassificationPredictorConfig::createLabelSpaceInfo(
            const IRowWiseLabelMatrix& labelMatrix) const {
        std::unique_ptr<LabelVectorSet> labelVectorSetPtr = std::make_unique<LabelVectorSet>();
        uint32 numRows = labelMatrix.getNumRows();

        for (uint32 i = 0; i < numRows; i++) {
            labelVectorSetPtr->addLabelVector(labelMatrix.createLabelVector(i));
        }

        return labelVectorSetPtr;
    }

}
