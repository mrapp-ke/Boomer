#include "boosting/output/predictor_regression_label_wise.hpp"
#include "common/validation.hpp"
#include "predictor_common.hpp"
#include "omp.h"


namespace boosting {

    LabelWiseRegressionPredictor::LabelWiseRegressionPredictor(uint32 numThreads)
        : numThreads_(numThreads) {
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    void LabelWiseRegressionPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                               CContiguousView<float64>& predictionMatrix, const Rule& rule,
                                               const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<float64>* predictionMatrixPtr = &predictionMatrix;
        const Rule* rulePtr = &rule;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(rulePtr) firstprivate(featureMatrixPtr) \
        firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            applyRule(*rulePtr, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i),
                      predictionMatrixPtr->row_begin(i));
        }
    }

    void LabelWiseRegressionPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                               CContiguousView<float64>& predictionMatrix, const Rule& rule,
                                               const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<float64>* predictionMatrixPtr = &predictionMatrix;
        const Rule* rulePtr = &rule;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(rulePtr) firstprivate(featureMatrixPtr) \
        firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            float32 tmpArray1[numFeatures];
            uint32 tmpArray2[numFeatures] = {};
            applyRuleCsr(*rulePtr, featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                         featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                         predictionMatrixPtr->row_begin(i), &tmpArray1[0], &tmpArray2[0], 1);
        }
    }

    void LabelWiseRegressionPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                               CContiguousView<float64>& predictionMatrix,
                                               const RuleModel& model, const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<float64>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
        firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                applyRule(rule, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i),
                          predictionMatrixPtr->row_begin(i));
            }
        }
    }

    void LabelWiseRegressionPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                               CContiguousView<float64>& predictionMatrix,
                                               const RuleModel& model, const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<float64>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
        firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            float32 tmpArray1[numFeatures];
            uint32 tmpArray2[numFeatures] = {};
            uint32 n = 1;

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                applyRuleCsr(rule, featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                             featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                             predictionMatrixPtr->row_begin(i), &tmpArray1[0], &tmpArray2[0], n);
                n++;
            }
        }
    }

}
