#include "boosting/losses/loss_label_wise.hpp"
#include "common/iterator/binary_forward_iterator.hpp"
#include "common/math/math.hpp"

#include <algorithm>

namespace boosting {

    /**
     * An implementation of the type `ILabelWiseLoss` that relies on an "update function" and an "evaluation function"
     * for updating the gradients and Hessians and evaluation the predictions for an individual label, respectively.
     */
    class LabelWiseLoss : virtual public ILabelWiseLoss {
        public:

            /**
             * A function that allows to update the gradient and Hessian for a single example and label. The function
             * accepts the true label, the predicted score, as well as pointers to the gradient and Hessian to be
             * updated, as arguments.
             */
            typedef void (*UpdateFunction)(bool trueLabel, float64 predictedScore, float64* gradient, float64* hessian);

            /**
             * A function that allows to calculate a numerical score that assesses the quality of the prediction for a
             * single example and label. The function accepts the true label and the predicted score as arguments and
             * returns a numerical score.
             */
            typedef float64 (*EvaluateFunction)(bool trueLabel, float64 predictedScore);

            /**
             * The "update function" that is used for updating gradients and Hessians.
             */
            const UpdateFunction updateFunction_;

            /**
             * The "evaluation function" that is used for evaluating predictions.
             */
            const EvaluateFunction evaluateFunction_;

            /**
             * @param updateFunction    The "update function" to be used for updating gradients and Hessians
             * @param evaluateFunction  The "evaluation function" to be used for evaluating predictions
             */
            LabelWiseLoss(UpdateFunction updateFunction, EvaluateFunction evaluateFunction)
                : updateFunction_(updateFunction), evaluateFunction_(evaluateFunction) {}

            void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                           const CContiguousConstView<float64>& scoreMatrix,
                                           CompleteIndexVector::const_iterator labelIndicesBegin,
                                           CompleteIndexVector::const_iterator labelIndicesEnd,
                                           DenseLabelWiseStatisticView& statisticView) const override final {
                DenseLabelWiseStatisticView::iterator statisticIterator = statisticView.begin(exampleIndex);
                CContiguousConstView<float64>::value_const_iterator scoreIterator =
                  scoreMatrix.values_cbegin(exampleIndex);
                CContiguousConstView<const uint8>::value_const_iterator labelIterator =
                  labelMatrix.values_cbegin(exampleIndex);
                uint32 numLabels = labelMatrix.getNumCols();

                for (uint32 i = 0; i < numLabels; i++) {
                    bool trueLabel = labelIterator[i];
                    float64 predictedScore = scoreIterator[i];
                    Tuple<float64>& tuple = statisticIterator[i];
                    (*updateFunction_)(trueLabel, predictedScore, &(tuple.first), &(tuple.second));
                }
            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                           const CContiguousConstView<float64>& scoreMatrix,
                                           PartialIndexVector::const_iterator labelIndicesBegin,
                                           PartialIndexVector::const_iterator labelIndicesEnd,
                                           DenseLabelWiseStatisticView& statisticView) const override final {
                DenseLabelWiseStatisticView::iterator statisticIterator = statisticView.begin(exampleIndex);
                CContiguousConstView<float64>::value_const_iterator scoreIterator =
                  scoreMatrix.values_cbegin(exampleIndex);
                CContiguousConstView<const uint8>::value_const_iterator labelIterator =
                  labelMatrix.values_cbegin(exampleIndex);
                uint32 numLabels = labelIndicesEnd - labelIndicesBegin;

                for (uint32 i = 0; i < numLabels; i++) {
                    uint32 labelIndex = labelIndicesBegin[i];
                    bool trueLabel = labelIterator[labelIndex];
                    float64 predictedScore = scoreIterator[labelIndex];
                    Tuple<float64>& tuple = statisticIterator[labelIndex];
                    (*updateFunction_)(trueLabel, predictedScore, &(tuple.first), &(tuple.second));
                }
            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                           const CContiguousConstView<float64>& scoreMatrix,
                                           CompleteIndexVector::const_iterator labelIndicesBegin,
                                           CompleteIndexVector::const_iterator labelIndicesEnd,
                                           DenseLabelWiseStatisticView& statisticView) const override final {
                DenseLabelWiseStatisticView::iterator statisticIterator = statisticView.begin(exampleIndex);
                CContiguousConstView<float64>::value_const_iterator scoreIterator =
                  scoreMatrix.values_cbegin(exampleIndex);
                auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                  labelMatrix.indices_cend(exampleIndex));
                uint32 numLabels = labelMatrix.getNumCols();

                for (uint32 i = 0; i < numLabels; i++) {
                    bool trueLabel = *labelIterator;
                    float64 predictedScore = scoreIterator[i];
                    Tuple<float64>& tuple = statisticIterator[i];
                    (*updateFunction_)(trueLabel, predictedScore, &(tuple.first), &(tuple.second));
                    labelIterator++;
                }
            }

            void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                           const CContiguousConstView<float64>& scoreMatrix,
                                           PartialIndexVector::const_iterator labelIndicesBegin,
                                           PartialIndexVector::const_iterator labelIndicesEnd,
                                           DenseLabelWiseStatisticView& statisticView) const override final {
                DenseLabelWiseStatisticView::iterator statisticIterator = statisticView.begin(exampleIndex);
                CContiguousConstView<float64>::value_const_iterator scoreIterator =
                  scoreMatrix.values_cbegin(exampleIndex);
                BinaryCsrConstView::index_const_iterator indexIterator = labelMatrix.indices_cbegin(exampleIndex);
                BinaryCsrConstView::index_const_iterator indicesEnd = labelMatrix.indices_cend(exampleIndex);
                uint32 numLabels = labelIndicesEnd - labelIndicesBegin;

                for (uint32 i = 0; i < numLabels; i++) {
                    uint32 labelIndex = labelIndicesBegin[i];
                    indexIterator = std::lower_bound(indexIterator, indicesEnd, labelIndex);
                    bool trueLabel = indexIterator != indicesEnd && *indexIterator == labelIndex;
                    float64 predictedScore = scoreIterator[labelIndex];
                    Tuple<float64>& tuple = statisticIterator[labelIndex];
                    (*updateFunction_)(trueLabel, predictedScore, &(tuple.first), &(tuple.second));
                }
            }

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                             const CContiguousConstView<float64>& scoreMatrix) const override final {
                CContiguousConstView<float64>::value_const_iterator scoreIterator =
                  scoreMatrix.values_cbegin(exampleIndex);
                CContiguousConstView<const uint8>::value_const_iterator labelIterator =
                  labelMatrix.values_cbegin(exampleIndex);
                uint32 numLabels = labelMatrix.getNumCols();
                float64 mean = 0;

                for (uint32 i = 0; i < numLabels; i++) {
                    float64 predictedScore = scoreIterator[i];
                    bool trueLabel = labelIterator[i];
                    float64 score = (*evaluateFunction_)(trueLabel, predictedScore);
                    mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
                }

                return mean;
            }

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                             const CContiguousConstView<float64>& scoreMatrix) const override final {
                CContiguousConstView<float64>::value_const_iterator scoreIterator =
                  scoreMatrix.values_cbegin(exampleIndex);
                auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                  labelMatrix.indices_cend(exampleIndex));
                uint32 numLabels = labelMatrix.getNumCols();
                float64 mean = 0;

                for (uint32 i = 0; i < numLabels; i++) {
                    float64 predictedScore = scoreIterator[i];
                    bool trueLabel = *labelIterator;
                    float64 score = (*evaluateFunction_)(trueLabel, predictedScore);
                    mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
                    labelIterator++;
                }

                return mean;
            }

            /**
             * @see `IDistanceMeasure::measureDistance`
             */
            float64 measureDistance(uint32 labelVectorIndex, const LabelVector& labelVector,
                                    VectorView<float64>::const_iterator scoresBegin,
                                    VectorView<float64>::const_iterator scoresEnd) const override final {
                uint32 numLabels = scoresEnd - scoresBegin;
                auto labelIterator = make_binary_forward_iterator(labelVector.cbegin(), labelVector.cend());
                float64 mean = 0;

                for (uint32 i = 0; i < numLabels; i++) {
                    float64 predictedScore = scoresBegin[i];
                    bool trueLabel = *labelIterator;
                    float64 score = (*evaluateFunction_)(trueLabel, predictedScore);
                    mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
                    labelIterator++;
                }

                return mean;
            }
    };

}
