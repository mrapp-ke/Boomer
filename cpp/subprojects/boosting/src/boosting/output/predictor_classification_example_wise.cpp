#include "boosting/output/predictor_classification_example_wise.hpp"
#include "common/validation.hpp"
#include "common/data/arrays.hpp"
#include "predictor_common.hpp"
#include "omp.h"
#include <algorithm>


namespace boosting {

    static inline const LabelVector* findClosestLabelVector(const float64* scoresBegin, const float64* scoresEnd,
                                                            const ISimilarityMeasure& measure,
                                                            const LabelVectorSet* labelVectorSet) {
        const LabelVector* closestLabelVector = nullptr;

        if (labelVectorSet != nullptr) {
            float64 bestScore = 0;
            uint32 bestCount = 0;

            for (auto it = labelVectorSet->cbegin(); it != labelVectorSet->cend(); it++) {
                const auto& entry = *it;
                const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
                uint32 count = entry.second;
                float64 score = measure.measureSimilarity(*labelVectorPtr, scoresBegin, scoresEnd);

                if (closestLabelVector == nullptr || score < bestScore || (score == bestScore && count > bestCount)) {
                    closestLabelVector = labelVectorPtr.get();
                    bestScore = score;
                    bestCount = count;
                }
            }
        }

        return closestLabelVector;
    }

    static inline void predictLabelVector(CContiguousView<uint8>::iterator predictionIterator, uint32 numElements,
                                          const LabelVector* labelVector) {
        setArrayToZeros(predictionIterator, numElements);

        if (labelVector != nullptr) {
            uint32 numIndices = labelVector->getNumElements();
            LabelVector::index_const_iterator indexIterator = labelVector->indices_cbegin();

            for (uint32 i = 0; i < numIndices; i++) {
                uint32 labelIndex = indexIterator[i];
                predictionIterator[labelIndex] = 1;
            }
        }
    }

    static inline uint32 predictLabelVector(LilMatrix<uint8>::Row& row, const LabelVector* labelVector) {
        uint32 numNonZeroElements = 0;

        if (labelVector != nullptr) {
            uint32 numIndices = labelVector->getNumElements();
            LabelVector::index_const_iterator indexIterator = labelVector->indices_cbegin();

            if (numIndices > 0) {
                uint32 labelIndex = indexIterator[0];
                row.emplace_front(labelIndex, 1);
                numNonZeroElements++;
                LilMatrix<uint8>::Row::iterator it = row.begin();

                for (uint32 i = 1; i < numIndices; i++) {
                    labelIndex = indexIterator[i];
                    it = row.emplace_after(it, labelIndex, 1);
                    numNonZeroElements++;
                }
            }
        }

        return numNonZeroElements;
    }

    ExampleWiseClassificationPredictor::ExampleWiseClassificationPredictor(
            std::unique_ptr<ISimilarityMeasure> measurePtr, uint32 numThreads)
        : measurePtr_(std::move(measurePtr)), numThreads_(numThreads) {
        assertNotNull("measurePtr", measurePtr_.get());
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    void ExampleWiseClassificationPredictor::transform(const CContiguousConstView<float64>& scoreMatrix,
                                                       CContiguousView<uint8>& predictionMatrix,
                                                       const LabelVectorSet* labelVectors) const {
        uint32 numExamples = scoreMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        const CContiguousConstView<float64>* scoreMatrixPtr = &scoreMatrix;
        CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
        const ISimilarityMeasure* measurePtr = measurePtr_.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(scoreMatrixPtr) \
        firstprivate(predictionMatrixPtr) firstprivate(measurePtr) firstprivate(labelVectors) schedule(dynamic) \
        num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            const LabelVector* closestLabelVector = findClosestLabelVector(scoreMatrixPtr->row_cbegin(i),
                                                                           scoreMatrixPtr->row_cend(i), *measurePtr,
                                                                           labelVectors);
            predictLabelVector(predictionMatrixPtr->row_begin(i), numLabels, closestLabelVector);
        }
    }

    void ExampleWiseClassificationPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                                     CContiguousView<uint8>& predictionMatrix,
                                                     const RuleModel& model, const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;
        const ISimilarityMeasure* measurePtr = measurePtr_.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
        firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) firstprivate(measurePtr) \
        firstprivate(labelVectors) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            float64 scoreVector[numLabels] = {};
            applyRules(*modelPtr, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i), &scoreVector[0]);
            const LabelVector* closestLabelVector = findClosestLabelVector(&scoreVector[0], &scoreVector[numLabels],
                                                                           *measurePtr, labelVectors);
            predictLabelVector(predictionMatrixPtr->row_begin(i), numLabels, closestLabelVector);
        }
    }

    void ExampleWiseClassificationPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                                     CContiguousView<uint8>& predictionMatrix,
                                                     const RuleModel& model, const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;
        const ISimilarityMeasure* measurePtr = measurePtr_.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
        firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) firstprivate(measurePtr) \
        firstprivate(labelVectors) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            float64 scoreVector[numLabels] = {};
            applyRulesCsr(*modelPtr, featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                          featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                          &scoreVector[0]);
            const LabelVector* closestLabelVector = findClosestLabelVector(&scoreVector[0], &scoreVector[numLabels],
                                                                           *measurePtr, labelVectors);
            predictLabelVector(predictionMatrixPtr->row_begin(i), numLabels, closestLabelVector);
        }
    }

    std::unique_ptr<SparsePredictionMatrix<uint8>> ExampleWiseClassificationPredictor::predict(
            const CContiguousFeatureMatrix& featureMatrix, uint32 numLabels, const RuleModel& model,
            const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        std::unique_ptr<LilMatrix<uint8>> lilMatrixPtr = std::make_unique<LilMatrix<uint8>>(numExamples);
        const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
        LilMatrix<uint8>* predictionMatrixPtr = lilMatrixPtr.get();
        const RuleModel* modelPtr = &model;
        const ISimilarityMeasure* measurePtr = measurePtr_.get();
        uint32 numNonZeroElements = 0;

        #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) firstprivate(numLabels) \
        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) \
        firstprivate(measurePtr) firstprivate(labelVectors) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            float64 scoreVector[numLabels] = {};
            applyRules(*modelPtr, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i), &scoreVector[0]);
            const LabelVector* closestLabelVector = findClosestLabelVector(&scoreVector[0], &scoreVector[numLabels],
                                                                           *measurePtr, labelVectors);
            numNonZeroElements += predictLabelVector(predictionMatrixPtr->getRow(i), closestLabelVector);
        }

        return std::make_unique<SparsePredictionMatrix<uint8>>(std::move(lilMatrixPtr), numLabels, numNonZeroElements);
    }

    std::unique_ptr<SparsePredictionMatrix<uint8>> ExampleWiseClassificationPredictor::predict(
            const CsrFeatureMatrix& featureMatrix, uint32 numLabels, const RuleModel& model,
            const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        std::unique_ptr<LilMatrix<uint8>> lilMatrixPtr = std::make_unique<LilMatrix<uint8>>(numExamples);
        const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
        LilMatrix<uint8>* predictionMatrixPtr = lilMatrixPtr.get();
        const RuleModel* modelPtr = &model;
        const ISimilarityMeasure* measurePtr = measurePtr_.get();
        uint32 numNonZeroElements = 0;

        #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) firstprivate(numLabels) \
        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) \
        firstprivate(measurePtr) firstprivate(labelVectors) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            float64 scoreVector[numLabels] = {};
            applyRulesCsr(*modelPtr, featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                          featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                          &scoreVector[0]);
            const LabelVector* closestLabelVector = findClosestLabelVector(&scoreVector[0], &scoreVector[numLabels],
                                                                           *measurePtr, labelVectors);
            numNonZeroElements += predictLabelVector(predictionMatrixPtr->getRow(i), closestLabelVector);
        }

        return std::make_unique<SparsePredictionMatrix<uint8>>(std::move(lilMatrixPtr), numLabels, numNonZeroElements);
    }

}
