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

            void updateExampleWiseStatistics(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                             const CContiguousView<float64>& scoreMatrix,
                                             DenseExampleWiseStatisticMatrix& statisticMatrix) const override;

            float64 evaluate(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                             const CContiguousView<float64>& scoreMatrix) const override;

            float64 measureSimilarity(const LabelVector& labelVector,
                                      CContiguousView<float64>::const_iterator scoresBegin,
                                      CContiguousView<float64>::const_iterator scoresEnd) const override;

    };

}
