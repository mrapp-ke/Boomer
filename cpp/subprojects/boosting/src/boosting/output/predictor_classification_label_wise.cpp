#include "boosting/output/predictor_classification_label_wise.hpp"
#include "predictor_common.hpp"
#include "omp.h"


namespace boosting {

    static inline void applyThreshold(CContiguousView<float64>::const_iterator originalIterator,
                                      CContiguousView<uint8>::iterator transformedIterator, uint32 numElements,
                                      float64 threshold) {
        for (uint32 i = 0; i < numElements; i++) {
            float64 originalValue = originalIterator[i];
            uint8 transformedValue = originalValue > threshold ? 1 : 0;
            transformedIterator[i] = transformedValue;
        }
    }

    LabelWiseClassificationPredictor::LabelWiseClassificationPredictor(float64 threshold, uint32 numThreads)
        : threshold_(threshold), numThreads_(numThreads) {

    }

    void LabelWiseClassificationPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                                   CContiguousView<uint8>& predictionMatrix,
                                                   const RuleModel& model) const {
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

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                applyRule(rule, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i), &scoreVector[0]);
            }

            applyThreshold(&scoreVector[0], predictionMatrixPtr->row_begin(i), numLabels, threshold_);
        }
    }

    void LabelWiseClassificationPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                                   CContiguousView<uint8>& predictionMatrix,
                                                   const RuleModel& model) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        uint32 numFeatures = featureMatrix.getNumCols();
        const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(numFeatures) \
        firstprivate(threshold_) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
        firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
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

            applyThreshold(&scoreVector[0], predictionMatrixPtr->row_begin(i), numLabels, threshold_);
        }
    }

}
