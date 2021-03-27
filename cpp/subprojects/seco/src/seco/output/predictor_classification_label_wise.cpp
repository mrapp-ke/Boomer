#include "seco/output/predictor_classification_label_wise.hpp"
#include "common/model/head_full.hpp"
#include "common/model/head_partial.hpp"
#include "omp.h"


namespace seco {

    static inline void applyFullHead(const FullHead& head, CContiguousView<uint8>::iterator begin,
                                     CContiguousView<uint8>::iterator end, CContiguousView<uint8>::iterator mask) {
        FullHead::score_const_iterator iterator = head.scores_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            if (mask[i] == 0) {
                uint8 prediction = iterator[i] > 0;
                begin[i] = prediction;
                mask[i] = 1;
            }
        }
    }

    static inline void applyPartialHead(const PartialHead& head, CContiguousView<uint8>::iterator begin,
                                        CContiguousView<uint8>::iterator end, CContiguousView<uint8>::iterator mask) {
        PartialHead::score_const_iterator scoreIterator = head.scores_cbegin();
        PartialHead::index_const_iterator indexIterator = head.indices_cbegin();
        uint32 numElements = head.getNumElements();

        for (uint32 i = 0; i < numElements; i++) {
            uint32 index = indexIterator[i];

            if (mask[index] == 0) {
                uint8 prediction = scoreIterator[i] > 0;
                begin[index] = prediction;
                mask[index] = 1;
            }
        }
    }

    static inline void applyHead(const IHead& head, CContiguousView<uint8>& predictionMatrix,
                                 CContiguousView<uint8>::iterator mask, uint32 row) {
        auto fullHeadVisitor = [&, row](const FullHead& head) {
            applyFullHead(head, predictionMatrix.row_begin(row), predictionMatrix.row_end(row), mask);
        };
        auto partialHeadVisitor = [&, row](const PartialHead& head) {
            applyPartialHead(head, predictionMatrix.row_begin(row), predictionMatrix.row_end(row), mask);
        };
        head.visit(fullHeadVisitor, partialHeadVisitor);
    }

    LabelWiseClassificationPredictor::LabelWiseClassificationPredictor(uint32 numThreads)
        : numThreads_(numThreads) {

    }

    void LabelWiseClassificationPredictor::predict(const CContiguousFeatureMatrix& featureMatrix,
                                                   CContiguousView<uint8>& predictionMatrix,
                                                   const RuleModel& model) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numLabels = predictionMatrix.getNumCols();
        const CContiguousFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
        firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            uint8 mask[numLabels] = {};

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                const IBody& body = rule.getBody();
                const IHead& head = rule.getHead();

                if (body.covers(featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i))) {
                    applyHead(head, *predictionMatrixPtr, &mask[0], i);
                }
            }
        }
    }

    void LabelWiseClassificationPredictor::predict(const CsrFeatureMatrix& featureMatrix,
                                                   CContiguousView<uint8>& predictionMatrix,
                                                   const RuleModel& model) const {
        uint32 numExamples = featureMatrix.getNumRows();
        uint32 numFeatures = featureMatrix.getNumCols();
        uint32 numLabels = predictionMatrix.getNumCols();
        const CsrFeatureMatrix* featureMatrixPtr = &featureMatrix;
        CContiguousView<uint8>* predictionMatrixPtr = &predictionMatrix;
        const RuleModel* modelPtr = &model;

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numFeatures) firstprivate(numLabels) \
        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) \
        num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            uint8 mask[numLabels] = {};
            float32 tmpArray1[numFeatures];
            uint32 tmpArray2[numFeatures] = {};
            uint32 n = 1;

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                const IBody& body = rule.getBody();
                const IHead& head = rule.getHead();

                if (body.covers(featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                                featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                                &tmpArray1[0], &tmpArray2[0], n)) {
                    applyHead(head, *predictionMatrixPtr, &mask[0], i);
                }

                n++;
            }
        }
    }

}
