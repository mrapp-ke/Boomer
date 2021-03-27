#include "boosting/output/predictor_classification_example_wise.hpp"
#include "predictor_common.hpp"
#include "omp.h"
#include <algorithm>


namespace boosting {

    template<class T>
    static inline void predictClosestLabelVector(uint32 exampleIndex, const float64* scoresBegin,
                                                 const float64* scoresEnd, CContiguousView<uint8>& predictionMatrix,
                                                 const ISimilarityMeasure& measure, const T& labelVectors) {
        std::fill(predictionMatrix.row_begin(exampleIndex), predictionMatrix.row_end(exampleIndex), 0);
        const LabelVector* closestLabelVector = nullptr;
        float64 bestScore = 0;
        uint32 bestCount = 0;

        for (auto it = labelVectors.cbegin(); it != labelVectors.cend(); it++) {
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

        if (closestLabelVector != nullptr) {
            CContiguousView<uint8>::iterator iterator = predictionMatrix.row_begin(exampleIndex);

            for (auto it = closestLabelVector->indices_cbegin(); it != closestLabelVector->indices_cend(); it++) {
                uint32 labelIndex = *it;
                iterator[labelIndex] = 1;
            }
        }
    }

    ExampleWiseClassificationPredictor::ExampleWiseClassificationPredictor(
            std::shared_ptr<ISimilarityMeasure> measurePtr, uint32 numThreads)
        : measurePtr_(measurePtr), numThreads_(numThreads) {

    }

    void ExampleWiseClassificationPredictor::addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr) {
        ++labelVectors_[std::move(labelVectorPtr)];
    }

    void ExampleWiseClassificationPredictor::visit(LabelVectorVisitor visitor) const {
        for (auto it = labelVectors_.cbegin(); it != labelVectors_.cend(); it++) {
            const auto& entry = *it;
            const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
            visitor(*labelVectorPtr);
        }
    }

    void ExampleWiseClassificationPredictor::transform(const CContiguousView<float64>& scoreMatrix,
                                                       CContiguousView<uint8>& predictionMatrix) const {
        uint32 numExamples = scoreMatrix.getNumRows();
        const CContiguousView<float64>* scoreMatrixPtr = &scoreMatrix;
        CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
        const ISimilarityMeasure* measurePtr = measurePtr_.get();
        const auto* labelVectorsPtr = &labelVectors_;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(scoreMatrixPtr) \
        firstprivate(predictionMatrixPtr) firstprivate(measurePtr) firstprivate(labelVectorsPtr) schedule(dynamic) \
        num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            predictClosestLabelVector(i, scoreMatrixPtr->row_cbegin(i), scoreMatrixPtr->row_cend(i),
                                      *predictionMatrixPtr, *measurePtr, *labelVectorsPtr);
        }
    }

    void ExampleWiseClassificationPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                                     CContiguousView<uint8>& predictionMatrix,
                                                     const RuleModel& model) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;
        const ISimilarityMeasure* measurePtr = measurePtr_.get();
        const auto* labelVectorsPtr = &labelVectors_;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
        firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) firstprivate(measurePtr) \
        firstprivate(labelVectorsPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            float64 scoreVector[numLabels] = {};

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                applyRule(rule, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i), &scoreVector[0]);
            }

            predictClosestLabelVector(i, &scoreVector[0], &scoreVector[numLabels], *predictionMatrixPtr, *measurePtr,
                                      *labelVectorsPtr);
        }
    }

    void ExampleWiseClassificationPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                                     CContiguousView<uint8>& predictionMatrix,
                                                     const RuleModel& model) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        uint32 numLabels = predictionMatrix.getNumCols();
        const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;
        const ISimilarityMeasure* measurePtr = measurePtr_.get();
        const auto* labelVectorsPtr = &labelVectors_;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
        firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) firstprivate(measurePtr) \
        firstprivate(labelVectorsPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            float64 scoreVector[numLabels] = {};
            float32 tmpArray1[numFeatures];
            uint32 tmpArray2[numFeatures] = {};
            uint32 n = 1;

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                applyRuleCsr(rule, featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                             featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                             &scoreVector[0], &tmpArray1[0], &tmpArray2[0], n);
                n++;
            }

            predictClosestLabelVector(i, &scoreVector[0], &scoreVector[numLabels], *predictionMatrixPtr, *measurePtr,
                                      *labelVectorsPtr);
        }
    }

}
