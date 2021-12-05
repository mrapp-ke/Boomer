/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/losses/loss_example_wise.hpp"


namespace boosting {

    /**
     * A multi-label variant of the logistic loss that is applied example-wise.
     */
    class ExampleWiseLogisticLoss final : public IExampleWiseLoss {

        public:

            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override;

            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override;

            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override;

            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                   const CContiguousConstView<float64> scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override;

            void updateExampleWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                             const CContiguousConstView<float64>& scoreMatrix,
                                             DenseExampleWiseStatisticView& statisticView) const override;

            void updateExampleWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                             const CContiguousConstView<float64>& scoreMatrix,
                                             DenseExampleWiseStatisticView& statisticView) const override;

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                             const CContiguousConstView<float64>& scoreMatrix) const override;

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                             const CContiguousConstView<float64>& scoreMatrix) const override;

            /**
             * @see `ISimilarityMeasure::measureSimilarity`
             */
            float64 measureSimilarity(const LabelVector& labelVector,
                                      CContiguousView<float64>::const_iterator scoresBegin,
                                      CContiguousView<float64>::const_iterator scoresEnd) const override;

    };

}
