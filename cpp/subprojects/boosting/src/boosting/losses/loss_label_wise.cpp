#include "boosting/losses/loss_label_wise.hpp"
#include "common/math/math.hpp"


namespace boosting {

    void AbstractLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex,
                                                          const IRandomAccessLabelMatrix& labelMatrix,
                                                          const CContiguousView<float64>& scoreMatrix,
                                                          FullIndexVector::const_iterator labelIndicesBegin,
                                                          FullIndexVector::const_iterator labelIndicesEnd,
                                                          DenseLabelWiseStatisticMatrix& statisticMatrix) const {
        DenseLabelWiseStatisticMatrix::gradient_iterator gradientIterator =
            statisticMatrix.gradients_row_begin(exampleIndex);
        DenseLabelWiseStatisticMatrix::hessian_iterator hessianIterator =
            statisticMatrix.hessians_row_begin(exampleIndex);
        CContiguousView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        uint32 numLabels = labelMatrix.getNumCols();

        for (uint32 i = 0; i < numLabels; i++) {
            bool trueLabel = labelMatrix.getValue(exampleIndex, i);
            float64 predictedScore = scoreIterator[i];
            this->updateGradientAndHessian(&gradientIterator[i], &hessianIterator[i], trueLabel, predictedScore);
        }
    }

    void AbstractLabelWiseLoss::updateLabelWiseStatistics(uint32 exampleIndex,
                                                          const IRandomAccessLabelMatrix& labelMatrix,
                                                          const CContiguousView<float64>& scoreMatrix,
                                                          PartialIndexVector::const_iterator labelIndicesBegin,
                                                          PartialIndexVector::const_iterator labelIndicesEnd,
                                                          DenseLabelWiseStatisticMatrix& statisticMatrix) const {
        DenseLabelWiseStatisticMatrix::gradient_iterator gradientIterator =
            statisticMatrix.gradients_row_begin(exampleIndex);
        DenseLabelWiseStatisticMatrix::hessian_iterator hessianIterator =
            statisticMatrix.hessians_row_begin(exampleIndex);
        CContiguousView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);

        for (auto indexIterator = labelIndicesBegin; indexIterator != labelIndicesEnd; indexIterator++) {
            uint32 labelIndex = *indexIterator;
            bool trueLabel = labelMatrix.getValue(exampleIndex, labelIndex);
            float64 predictedScore = scoreIterator[labelIndex];
            this->updateGradientAndHessian(&gradientIterator[labelIndex], &hessianIterator[labelIndex], trueLabel,
                                           predictedScore);
        }
    }

    float64 AbstractLabelWiseLoss::evaluate(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                            const CContiguousView<float64>& scoreMatrix) const {
        uint32 numLabels = labelMatrix.getNumCols();
        CContiguousView<float64>::const_iterator scoreIterator = scoreMatrix.row_cbegin(exampleIndex);
        float64 mean = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            bool trueLabel = labelMatrix.getValue(exampleIndex, i);
            float64 predictedScore = scoreIterator[i];
            float64 score = this->evaluate(trueLabel, predictedScore);
            mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
        }

        return mean;
    }

    float64 AbstractLabelWiseLoss::measureSimilarity(const LabelVector& labelVector,
                                                     CContiguousView<float64>::const_iterator scoresBegin,
                                                     CContiguousView<float64>::const_iterator scoresEnd) const {
        uint32 numLabels = scoresEnd - scoresBegin;
        LabelVector::index_const_iterator indexIterator = labelVector.indices_cbegin();
        LabelVector::index_const_iterator indicesEnd = labelVector.indices_cend();
        float64 mean = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoresBegin[i];
            bool trueLabel;

            if (indexIterator != indicesEnd && *indexIterator == i) {
                indexIterator++;
                trueLabel = true;
            } else {
                trueLabel = false;
            }

            float64 score = this->evaluate(trueLabel, predictedScore);
            mean = iterativeArithmeticMean<float64>(i + 1, score, mean);
        }

        return mean;
    }

}
