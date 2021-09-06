/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/losses/loss_label_wise.hpp"
#include "boosting/data/statistic_view_dense_example_wise.hpp"


namespace boosting {

    /**
     * Defines an interface for all (non-decomposable) loss functions that are applied example-wise.
     */
    class IExampleWiseLoss : public ILabelWiseLoss {

        public:

            virtual ~IExampleWiseLoss() { };

            /**
             * Updates the statistics of the example at a specific index.
             *
             * @param exampleIndex  The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix` that provides random
             *                      access to the labels of the training examples
             * @param scoreMatrix   A reference to an object of type `CContiguousConstView` that stores the currently
             *                      predicted scores
             * @param statisticView A reference to an object of type `DenseExampleWiseStatisticView` to be updated
             */
            virtual void updateExampleWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                     const CContiguousConstView<float64>& scoreMatrix,
                                                     DenseExampleWiseStatisticView& statisticView) const = 0;

             /**
             * Updates the statistics of the example at a specific index.
             *
             * @param exampleIndex  The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix   A reference to an object of type `CsrLabelMatrix` that provides row-wise access to
             *                      the labels of the training examples
             * @param scoreMatrix   A reference to an object of type `CContiguousConstView` that stores the currently
             *                      predicted scores
             * @param statisticView A reference to an object of type `DenseExampleWiseStatisticView` to be updated
             */
            virtual void updateExampleWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                     const CContiguousConstView<float64>& scoreMatrix,
                                                     DenseExampleWiseStatisticView& statisticView) const = 0;

    };

}
