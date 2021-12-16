#include "boosting/output/predictor_probability_label_wise.hpp"
#include "boosting/math/math.hpp"
#include "common/validation.hpp"
#include "predictor_common.hpp"
#include "omp.h"


namespace boosting {

    float64 LogisticFunction::transform(float64 predictedScore) const {
        return logisticFunction(predictedScore);
    }

    static inline void applyTransformationFunction(CContiguousConstView<float64>::const_iterator originalIterator,
                                                   CContiguousView<float64>::iterator transformedIterator,
                                                   uint32 numElements,
                                                   const ILabelWiseTransformationFunction& transformationFunction) {
        for (uint32 i = 0; i < numElements; i++) {
            float64 originalValue = originalIterator[i];
            float64 transformedValue = transformationFunction.transform(originalValue);
            transformedIterator[i] = transformedValue;
        }
    }

    LabelWiseProbabilityPredictor::LabelWiseProbabilityPredictor(
            std::unique_ptr<ILabelWiseTransformationFunction> transformationFunctionPtr, uint32 numThreads)
        : transformationFunctionPtr_(std::move(transformationFunctionPtr)), numThreads_(numThreads) {
        assertNotNull("transformationFunctionPtr", transformationFunctionPtr_.get());
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    void LabelWiseProbabilityPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                                CContiguousView<float64>& predictionMatrix,
                                                const RuleModel& model, const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<float64>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;
        const ILabelWiseTransformationFunction* transformationFunctionPtr = transformationFunctionPtr_.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
        firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) firstprivate(transformationFunctionPtr) \
        schedule(dynamic) num_threads(numThreads_)
        for (int64 i = 0; i < numExamples; i++) {
            float64* scoreVector = new float64[numLabels] {};

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                applyRule(rule, featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i), &scoreVector[0]);
            }

            applyTransformationFunction(&scoreVector[0], predictionMatrixPtr->row_begin(i), numLabels,
                                        *transformationFunctionPtr);
            delete[] scoreVector;
        }
    }

    void LabelWiseProbabilityPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                                CContiguousView<float64>& predictionMatrix,
                                                const RuleModel& model, const LabelVectorSet* labelVectors) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        uint32 numFeatures = featureMatrix.getNumCols();
        const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<float64>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;
        const ILabelWiseTransformationFunction* transformationFunctionPtr = transformationFunctionPtr_.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(numFeatures) \
        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) \
        firstprivate(transformationFunctionPtr) schedule(dynamic) num_threads(numThreads_)
        for (int64 i = 0; i < numExamples; i++) {
            float64* scoreVector = new float64[numLabels] {};
            float32* tmpArray1 = new float32[numFeatures];
            uint32* tmpArray2 = new uint32[numFeatures] {};
            uint32 n = 1;

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                applyRuleCsr(rule, featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                             featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                             &scoreVector[0], &tmpArray1[0], &tmpArray2[0], n);
                n++;
            }

            applyTransformationFunction(&scoreVector[0], predictionMatrixPtr->row_begin(i), numLabels,
                                        *transformationFunctionPtr);
            delete[] scoreVector;
            delete[] tmpArray1;
            delete[] tmpArray2;
        }
    }

}
