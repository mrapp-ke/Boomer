#include "boosting/output/predictor_classification_label_wise.hpp"
#include "common/validation.hpp"
#include "predictor_common.hpp"
#include "omp.h"


namespace boosting {

    static inline void applyThreshold(CContiguousConstView<float64>::const_iterator originalIterator,
                                      CContiguousView<uint8>::iterator transformedIterator, uint32 numElements,
                                      float64 threshold) {
        for (uint32 i = 0; i < numElements; i++) {
            float64 originalValue = originalIterator[i];
            uint8 transformedValue = originalValue > threshold ? 1 : 0;
            transformedIterator[i] = transformedValue;
        }
    }

    static inline uint32 applyThreshold(CContiguousConstView<float64>::const_iterator originalIterator,
                                        LilMatrix<uint8>::Row& row, uint32 numElements, float64 threshold) {
        uint32 numNonZeroElements = 0;
        uint32 i = 0;

        for (; i < numElements; i++) {
            float64 originalValue = originalIterator[i];

            if (originalValue > threshold) {
                row.emplace_front(i, 1);
                numNonZeroElements++;
                break;
            }
        }

        LilMatrix<uint8>::Row::iterator it = row.begin();

        for (i = i + 1; i < numElements; i++) {
            float64 originalValue = originalIterator[i];

            if (originalValue > threshold) {
                it = row.emplace_after(it, i, 1);
                numNonZeroElements++;
            }
        }

        return numNonZeroElements;
    }

    LabelWiseClassificationPredictor::LabelWiseClassificationPredictor(float64 threshold, uint32 numThreads)
        : threshold_(threshold), numThreads_(numThreads) {
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    void LabelWiseClassificationPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                                   CContiguousView<uint8>& predictionMatrix,
                                                   const RuleModel& model, const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(threshold_) \
        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) \
        num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            float64 scoreVector[numLabels] = {};
            applyRules(*modelPtr, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i), &scoreVector[0]);
            applyThreshold(&scoreVector[0], predictionMatrixPtr->row_begin(i), numLabels, threshold_);
        }
    }

    void LabelWiseClassificationPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                                   CContiguousView<uint8>& predictionMatrix,
                                                   const RuleModel& model, const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        uint32 numLabels = predictionMatrix.getNumCols();
        const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) firstprivate(numLabels) \
        firstprivate(threshold_) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
        firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            float64 scoreVector[numLabels] = {};
            applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                          featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                          featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
            applyThreshold(&scoreVector[0], predictionMatrixPtr->row_begin(i), numLabels, threshold_);
        }
    }

    std::unique_ptr<SparsePredictionMatrix<uint8>> LabelWiseClassificationPredictor::predict(
            const CContiguousFeatureMatrix& featureMatrix, uint32 numLabels, const RuleModel& model,
            const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        std::unique_ptr<LilMatrix<uint8>> lilMatrixPtr = std::make_unique<LilMatrix<uint8>>(numExamples);
        const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
        LilMatrix<uint8>* predictionMatrixPtr = lilMatrixPtr.get();
        const RuleModel* modelPtr = &model;
        uint32 numNonZeroElements = 0;

        #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) firstprivate(numLabels) \
        firstprivate(threshold_) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
        firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            float64 scoreVector[numLabels] = {};
            applyRules(*modelPtr, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i), &scoreVector[0]);
            numNonZeroElements += applyThreshold(&scoreVector[0], predictionMatrixPtr->getRow(i), numLabels,
                                                 threshold_);
        }

        return std::make_unique<SparsePredictionMatrix<uint8>>(std::move(lilMatrixPtr), numLabels, numNonZeroElements);
    }

    std::unique_ptr<SparsePredictionMatrix<uint8>> LabelWiseClassificationPredictor::predict(
            const CsrFeatureMatrix& featureMatrix, uint32 numLabels, const RuleModel& model,
            const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        std::unique_ptr<LilMatrix<uint8>> lilMatrixPtr = std::make_unique<LilMatrix<uint8>>(numExamples);
        const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
        LilMatrix<uint8>* predictionMatrixPtr = lilMatrixPtr.get();
        const RuleModel* modelPtr = &model;
        uint32 numNonZeroElements = 0;

        #pragma omp parallel for reduction(+:numNonZeroElements) firstprivate(numExamples) firstprivate(numFeatures) \
        firstprivate(numLabels) firstprivate(threshold_) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
        firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            float64 scoreVector[numLabels] = {};
            applyRulesCsr(*modelPtr, numFeatures, featureMatrixPtr->row_indices_cbegin(i),
                          featureMatrixPtr->row_indices_cend(i), featureMatrixPtr->row_values_cbegin(i),
                          featureMatrixPtr->row_values_cend(i), &scoreVector[0]);
            numNonZeroElements += applyThreshold(&scoreVector[0], predictionMatrixPtr->getRow(i), numLabels,
                                                 threshold_);
        }

        return std::make_unique<SparsePredictionMatrix<uint8>>(std::move(lilMatrixPtr), numLabels, numNonZeroElements);
    }

}
