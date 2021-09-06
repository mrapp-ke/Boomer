#include "boosting/losses/loss_label_wise.hpp"
#include "common/math/math.hpp"


namespace boosting {

    AbstractLabelWiseLoss::AbstractLabelWiseLoss(UpdateFunction updateFunction, EvaluateFunction evaluateFunction)
        : updateFunction_(updateFunction), evaluateFunction_(evaluateFunction) {

    }

    void AbstractLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex,
                                                          const CContiguousLabelMatrix& labelMatrix,
                                                          const CContiguousConstView<float64>& scoreMatrix,
                                                          CompleteIndexVector::const_iterator labelIndicesBegin,
                                                          CompleteIndexVector::const_iterator labelIndicesEnd,
                                                          DenseLabelWiseStatisticView& statisticView) const {
        DenseLabelWiseStatisticView::iterator statisticIterator = statisticView.row_begin(exampleIndex);
        CContiguousConstView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        uint32 numLabels = labelMatrix.getNumCols();

        for (uint32 i = 0; i < numLabels; i++) {
            bool trueLabel = labelIterator[i];
            float64 predictedScore = scoreIterator[i];
            Tuple<float64>& tuple = statisticIterator[i];
            (*updateFunction_)(trueLabel, predictedScore, &(tuple.first), &(tuple.second));
        }
    }

    void AbstractLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex,
                                                          const CContiguousLabelMatrix& labelMatrix,
                                                          const CContiguousConstView<float64>& scoreMatrix,
                                                          PartialIndexVector::const_iterator labelIndicesBegin,
                                                          PartialIndexVector::const_iterator labelIndicesEnd,
                                                          DenseLabelWiseStatisticView& statisticView) const {
        DenseLabelWiseStatisticView::iterator statisticIterator = statisticView.row_begin(exampleIndex);
        CContiguousConstView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        uint32 numLabels = labelIndicesEnd - labelIndicesBegin;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 labelIndex = labelIndicesBegin[i];
            bool trueLabel = labelIterator[labelIndex];
            float64 predictedScore = scoreIterator[labelIndex];
            Tuple<float64>& tuple = statisticIterator[labelIndex];
            (*updateFunction_)(trueLabel, predictedScore, &(tuple.first), &(tuple.second));
        }
    }

    void AbstractLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex,
                                                          const CsrLabelMatrix& labelMatrix,
                                                          const CContiguousConstView<float64>& scoreMatrix,
                                                          CompleteIndexVector::const_iterator labelIndicesBegin,
                                                          CompleteIndexVector::const_iterator labelIndicesEnd,
                                                          DenseLabelWiseStatisticView& statisticView) const {
        DenseLabelWiseStatisticView::iterator statisticIterator = statisticView.row_begin(exampleIndex);
        CContiguousConstView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        CsrLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        uint32 numLabels = labelMatrix.getNumCols();

        for (uint32 i = 0; i < numLabels; i++) {
            bool trueLabel = *labelIterator;
            float64 predictedScore = scoreIterator[i];
            Tuple<float64>& tuple = statisticIterator[i];
            (*updateFunction_)(trueLabel, predictedScore, &(tuple.first), &(tuple.second));
            labelIterator++;
        }
    }

    void AbstractLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                          const CContiguousConstView<float64> scoreMatrix,
                                                          PartialIndexVector::const_iterator labelIndicesBegin,
                                                          PartialIndexVector::const_iterator labelIndicesEnd,
                                                          DenseLabelWiseStatisticView& statisticView) const {
        DenseLabelWiseStatisticView::iterator statisticIterator = statisticView.row_begin(exampleIndex);
        CContiguousConstView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        CsrLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        uint32 numLabels = labelIndicesEnd - labelIndicesBegin;
        uint32 previousLabelIndex = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 labelIndex = labelIndicesBegin[i];
            std::advance(labelIterator, labelIndex - previousLabelIndex);
            bool trueLabel = *labelIterator;
            float64 predictedScore = scoreIterator[labelIndex];
            Tuple<float64>& tuple = statisticIterator[labelIndex];
            (*updateFunction_)(trueLabel, predictedScore, &(tuple.first), &(tuple.second));
            previousLabelIndex = labelIndex;
        }
    }

    float64 AbstractLabelWiseLoss::evaluate(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                            const CContiguousConstView<float64>& scoreMatrix) const {
        CContiguousConstView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
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

    float64 AbstractLabelWiseLoss::evaluate(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                            const CContiguousConstView<float64>& scoreMatrix) const {
        CContiguousConstView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        CsrLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
        uint32 numLabels = labelMatrix.getNumCols();
        float64 mean = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel= *labelIterator;
            float64 score = (*evaluateFunction_)(trueLabel, predictedScore);
            mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
            labelIterator++;
        }

        return mean;
    }

    float64 AbstractLabelWiseLoss::measureSimilarity(const LabelVector& labelVector,
                                                     CContiguousConstView<float64>::const_iterator scoresBegin,
                                                     CContiguousConstView<float64>::const_iterator scoresEnd) const {
        uint32 numLabels = scoresEnd - scoresBegin;
        auto labelIterator = make_binary_forward_iterator(labelVector.indices_cbegin(), labelVector.indices_cend());
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

}
