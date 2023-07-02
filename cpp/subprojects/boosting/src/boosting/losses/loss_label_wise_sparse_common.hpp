/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "boosting/losses/loss_label_wise_sparse.hpp"
#include "common/iterator/non_zero_index_forward_iterator.hpp"
#include "loss_label_wise_common.hpp"

#include <algorithm>

namespace boosting {

    static const uint32 LIMIT = std::numeric_limits<uint32>::max();

    template<typename IndexIterator>
    static inline uint32 fetchNextStatistic(IndexIterator& indexIterator, IndexIterator indicesEnd,
                                            SparseSetMatrix<float64>::const_iterator& scoreIterator,
                                            SparseSetMatrix<float64>::const_iterator scoresEnd, Tuple<float64>& tuple,
                                            LabelWiseLoss::UpdateFunction updateFunction) {
        uint32 labelIndex = indexIterator == indicesEnd ? LIMIT : *indexIterator;
        uint32 scoreIndex = scoreIterator == scoresEnd ? LIMIT : (*scoreIterator).index;

        if (scoreIndex < labelIndex) {
            (*updateFunction)(false, (*scoreIterator).value, &tuple.first, &tuple.second);
            scoreIterator++;
            return scoreIndex;
        } else if (labelIndex < scoreIndex) {
            (*updateFunction)(true, 0, &tuple.first, &tuple.second);
            indexIterator++;
            return labelIndex;
        } else if (labelIndex < LIMIT) {
            (*updateFunction)(true, (*scoreIterator).value, &tuple.first, &tuple.second);
            scoreIterator++;
            indexIterator++;
            return labelIndex;
        }

        return LIMIT;
    }

    template<typename IndexIterator>
    static inline uint32 fetchNextNonZeroStatistic(IndexIterator& indexIterator, IndexIterator indicesEnd,
                                                   SparseSetMatrix<float64>::const_iterator& scoreIterator,
                                                   SparseSetMatrix<float64>::const_iterator scoresEnd,
                                                   Tuple<float64>& tuple,
                                                   LabelWiseLoss::UpdateFunction updateFunction) {
        uint32 index = fetchNextStatistic(indexIterator, indicesEnd, scoreIterator, scoresEnd, tuple, updateFunction);

        while (tuple.first == 0 && index < LIMIT) {
            index = fetchNextStatistic(indexIterator, indicesEnd, scoreIterator, scoresEnd, tuple, updateFunction);
        }

        return index;
    }

    template<typename IndexIterator>
    static inline void updateLabelWiseStatisticsInternally(IndexIterator indicesBegin, IndexIterator indicesEnd,
                                                           SparseSetMatrix<float64>::const_iterator scoresBegin,
                                                           SparseSetMatrix<float64>::const_iterator scoresEnd,
                                                           SparseLabelWiseStatisticView::row row,
                                                           LabelWiseLoss::UpdateFunction updateFunction) {
        row.clear();
        Tuple<float64> tuple;
        uint32 index;

        while (
          (index = fetchNextNonZeroStatistic(indicesBegin, indicesEnd, scoresBegin, scoresEnd, tuple, updateFunction))
          < LIMIT) {
            IndexedValue<Tuple<float64>>& entry = row.emplace(index);
            entry.value = tuple;
        }
    }

    template<typename IndexIterator>
    static inline uint32 fetchNextEvaluation(IndexIterator& indexIterator, IndexIterator indicesEnd,
                                             SparseSetMatrix<float64>::const_iterator& scoreIterator,
                                             SparseSetMatrix<float64>::const_iterator scoresEnd, float64& score,
                                             LabelWiseLoss::EvaluateFunction evaluateFunction) {
        uint32 labelIndex = indexIterator == indicesEnd ? LIMIT : *indexIterator;
        uint32 scoreIndex = scoreIterator == scoresEnd ? LIMIT : (*scoreIterator).index;

        if (scoreIndex < labelIndex) {
            score = (*evaluateFunction)(false, (*scoreIterator).value);
            scoreIterator++;
            return scoreIndex;
        } else if (labelIndex < scoreIndex) {
            score = (*evaluateFunction)(true, 0);
            indexIterator++;
            return labelIndex;
        } else if (labelIndex < LIMIT) {
            score = (*evaluateFunction)(true, (*scoreIterator).value);
            scoreIterator++;
            indexIterator++;
            return labelIndex;
        }

        return LIMIT;
    }

    template<typename IndexIterator>
    static inline uint32 fetchNextNonZeroEvaluation(IndexIterator& indexIterator, IndexIterator indicesEnd,
                                                    SparseSetMatrix<float64>::const_iterator& scoreIterator,
                                                    SparseSetMatrix<float64>::const_iterator scoresEnd, float64& score,
                                                    LabelWiseLoss::EvaluateFunction evaluateFunction) {
        uint32 index =
          fetchNextEvaluation(indexIterator, indicesEnd, scoreIterator, scoresEnd, score, evaluateFunction);

        while (score == 0 && index < LIMIT) {
            index = fetchNextEvaluation(indexIterator, indicesEnd, scoreIterator, scoresEnd, score, evaluateFunction);
        }

        return index;
    }

    template<typename IndexIterator>
    static inline float64 evaluateInternally(IndexIterator indicesBegin, IndexIterator indicesEnd,
                                             SparseSetMatrix<float64>::const_iterator scoresBegin,
                                             SparseSetMatrix<float64>::const_iterator scoresEnd,
                                             LabelWiseLoss::EvaluateFunction evaluateFunction, uint32 numLabels) {
        float64 mean = 0;
        float64 score = 0;
        uint32 i = 0;

        while (fetchNextNonZeroEvaluation(indicesBegin, indicesEnd, scoresBegin, scoresEnd, score, evaluateFunction)
               < LIMIT) {
            mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
            i++;
        }

        return mean * ((float64) i / (float64) numLabels);
    }

    /**
     * An implementation of the type `ISparseLabelWiseLoss` that relies on an "update function" and an
     * "evaluation function" for updating the gradients and Hessians and evaluation the predictions for an individual
     * label, respectively.
     */
    class SparseLabelWiseLoss final : public LabelWiseLoss,
                                      public ISparseLabelWiseLoss {
        public:

            /**
             * @param updateFunction    The "update function" to be used for updating gradients and Hessians
             * @param evaluateFunction  The "evaluation function" to be used for evaluating predictions
             */
            SparseLabelWiseLoss(UpdateFunction updateFunction, EvaluateFunction evaluateFunction)
                : LabelWiseLoss(updateFunction, evaluateFunction) {}

            /**
             * Keep "updateLabelWiseStatistics" functions from the parent class rather than hiding them.
             */
            using LabelWiseLoss::updateLabelWiseStatistics;

            void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                           const SparseSetMatrix<float64>& scoreMatrix,
                                           CompleteIndexVector::const_iterator labelIndicesBegin,
                                           CompleteIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override {
                auto indicesBegin = make_non_zero_index_forward_iterator(labelMatrix.values_cbegin(exampleIndex),
                                                                         labelMatrix.values_cend(exampleIndex));
                auto indicesEnd = make_non_zero_index_forward_iterator(labelMatrix.values_cend(exampleIndex),
                                                                       labelMatrix.values_cend(exampleIndex));
                updateLabelWiseStatisticsInternally(indicesBegin, indicesEnd, scoreMatrix.cbegin(exampleIndex),
                                                    scoreMatrix.cend(exampleIndex), statisticView[exampleIndex],
                                                    LabelWiseLoss::updateFunction_);
            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                           const SparseSetMatrix<float64>& scoreMatrix,
                                           PartialIndexVector::const_iterator labelIndicesBegin,
                                           PartialIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override {
                const SparseSetMatrix<float64>::const_row scoreMatrixRow = scoreMatrix[exampleIndex];
                CContiguousConstView<const uint8>::value_const_iterator labelIterator =
                  labelMatrix.values_cbegin(exampleIndex);
                SparseLabelWiseStatisticView::row statisticViewRow = statisticView[exampleIndex];
                uint32 numElements = labelIndicesEnd - labelIndicesBegin;
                Tuple<float64> tuple;

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = labelIndicesBegin[i];
                    const IndexedValue<float64>* scoreMatrixEntry = scoreMatrixRow[index];
                    float64 predictedScore = scoreMatrixEntry ? scoreMatrixEntry->value : 0;
                    bool trueLabel = labelIterator[index];
                    (*LabelWiseLoss::updateFunction_)(trueLabel, predictedScore, &tuple.first, &tuple.second);

                    if (tuple.first != 0) {
                        IndexedValue<Tuple<float64>>& statisticViewEntry = statisticViewRow.emplace(index);
                        statisticViewEntry.value = tuple;
                    } else {
                        statisticViewRow.erase(index);
                    }
                }
            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                           const SparseSetMatrix<float64>& scoreMatrix,
                                           CompleteIndexVector::const_iterator labelIndicesBegin,
                                           CompleteIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override {
                updateLabelWiseStatisticsInternally(labelMatrix.indices_cbegin(exampleIndex),
                                                    labelMatrix.indices_cend(exampleIndex),
                                                    scoreMatrix.cbegin(exampleIndex), scoreMatrix.cend(exampleIndex),
                                                    statisticView[exampleIndex], LabelWiseLoss::updateFunction_);
            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                           const SparseSetMatrix<float64>& scoreMatrix,
                                           PartialIndexVector::const_iterator labelIndicesBegin,
                                           PartialIndexVector::const_iterator labelIndicesEnd,
                                           SparseLabelWiseStatisticView& statisticView) const override {
                const SparseSetMatrix<float64>::const_row scoreMatrixRow = scoreMatrix[exampleIndex];
                BinaryCsrConstView::index_const_iterator indexIterator = labelMatrix.indices_cbegin(exampleIndex);
                BinaryCsrConstView::index_const_iterator indicesEnd = labelMatrix.indices_cend(exampleIndex);
                SparseLabelWiseStatisticView::row statisticViewRow = statisticView[exampleIndex];
                uint32 numElements = labelIndicesEnd - labelIndicesBegin;
                Tuple<float64> tuple;

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = labelIndicesBegin[i];
                    indexIterator = std::lower_bound(indexIterator, indicesEnd, index);
                    bool trueLabel = indexIterator != indicesEnd && *indexIterator == index;
                    const IndexedValue<float64>* scoreMatrixEntry = scoreMatrixRow[index];
                    float64 predictedScore = scoreMatrixEntry ? scoreMatrixEntry->value : 0;
                    (*LabelWiseLoss::updateFunction_)(trueLabel, predictedScore, &tuple.first, &tuple.second);

                    if (tuple.first != 0) {
                        IndexedValue<Tuple<float64>>& statisticViewEntry = statisticViewRow.emplace(index);
                        statisticViewEntry.value = tuple;
                    } else {
                        statisticViewRow.erase(index);
                    }
                }
            }

            // Keep "evaluate" functions from the parent class rather than hiding them
            using LabelWiseLoss::evaluate;

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                             const SparseSetMatrix<float64>& scoreMatrix) const override {
                auto indicesBegin = make_non_zero_index_forward_iterator(labelMatrix.values_cbegin(exampleIndex),
                                                                         labelMatrix.values_cend(exampleIndex));
                auto indicesEnd = make_non_zero_index_forward_iterator(labelMatrix.values_cend(exampleIndex),
                                                                       labelMatrix.values_cend(exampleIndex));
                return evaluateInternally(indicesBegin, indicesEnd, scoreMatrix.cbegin(exampleIndex),
                                          scoreMatrix.cend(exampleIndex), LabelWiseLoss::evaluateFunction_,
                                          labelMatrix.getNumCols());
            }

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                             const SparseSetMatrix<float64>& scoreMatrix) const override {
                return evaluateInternally(labelMatrix.indices_cbegin(exampleIndex),
                                          labelMatrix.indices_cend(exampleIndex), scoreMatrix.cbegin(exampleIndex),
                                          scoreMatrix.cend(exampleIndex), LabelWiseLoss::evaluateFunction_,
                                          labelMatrix.getNumCols());
            }
    };

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
