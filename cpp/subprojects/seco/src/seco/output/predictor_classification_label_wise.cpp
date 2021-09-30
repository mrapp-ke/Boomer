#include "seco/output/predictor_classification_label_wise.hpp"
#include "common/iterator/non_zero_index_forward_iterator.hpp"
#include "common/model/head_complete.hpp"
#include "common/model/head_partial.hpp"
#include "common/validation.hpp"
#include "omp.h"


namespace seco {

    static inline void applyCompleteHead(const CompleteHead& head, CContiguousView<uint8>::iterator begin,
                                         CContiguousView<uint8>::iterator end, CContiguousView<uint8>::iterator mask) {
        CompleteHead::score_const_iterator iterator = head.scores_cbegin();
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
        auto completeHeadVisitor = [&, row](const CompleteHead& head) {
            applyCompleteHead(head, predictionMatrix.row_begin(row), predictionMatrix.row_end(row), mask);
        };
        auto partialHeadVisitor = [&, row](const PartialHead& head) {
            applyPartialHead(head, predictionMatrix.row_begin(row), predictionMatrix.row_end(row), mask);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
    }

    template<typename ScoreIterator, typename IndexIterator>
    static inline uint32 addFirst(ScoreIterator& scoresBegin, ScoreIterator scoresEnd, IndexIterator indexIterator,
                                  LilMatrix<uint8>::Row& row) {
        if (scoresBegin != scoresEnd) {
            uint32 index = indexIterator[*scoresBegin];

            if (row.empty()) {
                row.emplace_front(index, 1);
                scoresBegin++;
                return 1;
            } else {
                LilMatrix<uint8>::Row::iterator it = row.begin();
                uint32 firstIndex = (*it).index;

                if (index == firstIndex) {
                    scoresBegin++;
                } else if (index < firstIndex) {
                    row.emplace_front(index, 1);
                    scoresBegin++;
                    return 1;
                }
            }
        }

        return 0;
    }

    template<typename ScoreIterator, typename IndexIterator>
    static inline uint32 applyHead(ScoreIterator scoresBegin, ScoreIterator scoresEnd, IndexIterator indexIterator,
                                   LilMatrix<uint8>::Row& row) {
        uint32 numNonZeroElements = addFirst(scoresBegin, scoresEnd, indexIterator, row);
        LilMatrix<uint8>::Row::iterator prevIt = row.begin();
        LilMatrix<uint8>::Row::iterator it = prevIt;
        it++;

        for (; scoresBegin != scoresEnd && it != row.end(); scoresBegin++) {
            uint32 index = indexIterator[*scoresBegin];
            uint32 currentIndex = (*it).index;
            LilMatrix<uint8>::Row::iterator nextIt = it;
            nextIt++;

            while (index > currentIndex && nextIt != row.end()) {
                uint32 nextIndex = (*nextIt).index;

                if (index >= nextIndex) {
                    currentIndex = nextIndex;
                    prevIt = it;
                    it = nextIt;
                    nextIt++;
                } else {
                    break;
                }
            }

            if (index > currentIndex) {
                prevIt = row.emplace_after(it, index, 1);
                numNonZeroElements++;
                it = prevIt;
            } else if (index < currentIndex) {
                prevIt = row.emplace_after(prevIt, index, 1);
                numNonZeroElements++;
                it = prevIt;
            } else {
                prevIt = it;
            }

            it++;
        }

        for (; scoresBegin != scoresEnd; scoresBegin++) {
            uint32 index = indexIterator[*scoresBegin];
            prevIt = row.emplace_after(prevIt, index, 1);
            numNonZeroElements++;
        }

        return numNonZeroElements;
    }

    static inline uint32 applyHead(const IHead& head, LilMatrix<uint8>::Row& row) {
        uint32 numNonZeroElements;
        auto completeHeadVisitor = [&](const CompleteHead& head) mutable {
            numNonZeroElements = applyHead(
                make_non_zero_index_forward_iterator(head.scores_cbegin(), head.scores_cend()),
                make_non_zero_index_forward_iterator(head.scores_cend(), head.scores_cend()), IndexIterator(0), row);
        };
        auto partialHeadVisitor = [&](const PartialHead& head) mutable {
            numNonZeroElements = applyHead(
                make_non_zero_index_forward_iterator(head.scores_cbegin(), head.scores_cend()),
                make_non_zero_index_forward_iterator(head.scores_cend(), head.scores_cend()), head.indices_cbegin(),
                row);
        };
        head.visit(completeHeadVisitor, partialHeadVisitor);
        return numNonZeroElements;
    }

    LabelWiseClassificationPredictor::LabelWiseClassificationPredictor(uint32 numThreads)
        : numThreads_(numThreads) {
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

        #pragma omp parallel for firstprivate(numExamples) firstprivate(numLabels) firstprivate(modelPtr) \
        firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            uint8 mask[numLabels] = {};

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                const IBody& body = rule.getBody();

                if (body.covers(featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i))) {
                    const IHead& head = rule.getHead();
                    applyHead(head, *predictionMatrixPtr, &mask[0], i);
                }
            }
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

                if (body.covers(featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                                featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                                &tmpArray1[0], &tmpArray2[0], n)) {
                    const IHead& head = rule.getHead();
                    applyHead(head, *predictionMatrixPtr, &mask[0], i);
                }

                n++;
            }
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
        firstprivate(modelPtr) firstprivate(featureMatrixPtr) firstprivate(predictionMatrixPtr) schedule(dynamic) \
        num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            LilMatrix<uint8>::Row& row = predictionMatrixPtr->getRow(i);

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                const IBody& body = rule.getBody();

                if (body.covers(featureMatrixPtr->row_cbegin(i), featureMatrixPtr->row_cend(i))) {
                    const IHead& head = rule.getHead();
                    numNonZeroElements += applyHead(head, row);
                }
            }
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
        firstprivate(numLabels) firstprivate(modelPtr) firstprivate(featureMatrixPtr) \
        firstprivate(predictionMatrixPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            LilMatrix<uint8>::Row& row = predictionMatrixPtr->getRow(i);
            float32 tmpArray1[numFeatures];
            uint32 tmpArray2[numFeatures] = {};
            uint32 n = 1;

            for (auto it = modelPtr->used_cbegin(); it != modelPtr->used_cend(); it++) {
                const Rule& rule = *it;
                const IBody& body = rule.getBody();

                if (body.covers(featureMatrixPtr->row_indices_cbegin(i), featureMatrixPtr->row_indices_cend(i),
                                featureMatrixPtr->row_values_cbegin(i), featureMatrixPtr->row_values_cend(i),
                                &tmpArray1[0], &tmpArray2[0], n)) {
                    const IHead& head = rule.getHead();
                    numNonZeroElements += applyHead(head, row);
                }

                n++;
            }
        }

        return std::make_unique<SparsePredictionMatrix<uint8>>(std::move(lilMatrixPtr), numLabels, numNonZeroElements);
    }

}
