/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/binning/label_binning.hpp"


namespace boosting {

    /**
     * Assigns labels to bins, based on the corresponding gradients and Hessians, in a way such that each bin contains
     * labels for which the predicted score is expected to belong to the same value range.
     *
     * @tparam GradientIterator The type of the iterator that provides access to the gradients
     * @tparam HessianIterator  The type of the iterator that provides access to the Hessians
     */
    template<class GradientIterator, class HessianIterator>
    class EqualWidthLabelBinning final : public ILabelBinning<GradientIterator, HessianIterator> {

        private:

            float32 binRatio_;

            uint32 minBins_;

            uint32 maxBins_;

        public:

            /**
             * @param binRatio  A percentage that specifies how many bins should be used to assign labels to, e.g., if
             *                  100 labels are available, 0.5 means that `ceil(0.5 * 100) = 50` bins should be used
             * @param minBins   The minimum number of bins to be used to assign labels to. Must be at least 2
             * @param maxBins   The maximum number of bins to be used to assign labels to. Must be at least `minBins` or
             *                  0, if the maximum number of bins should not be restricted
             */
            EqualWidthLabelBinning(float32 binRatio, uint32 minBins, uint32 maxBins);

            uint32 getMaxBins(uint32 numLabels) const override;

            LabelInfo getLabelInfo(GradientIterator gradientsBegin, GradientIterator gradientsEnd,
                                   HessianIterator hessiansBegin, HessianIterator hessiansEnd,
                                   float64 l2RegularizationWeight) const override;

            void createBins(
                LabelInfo labelInfo, GradientIterator gradientsBegin, GradientIterator gradientsEnd,
                HessianIterator hessiansBegin, HessianIterator hessiansEnd, float64 l2RegularizationWeight,
                typename ILabelBinning<GradientIterator, HessianIterator>::Callback callback,
                typename ILabelBinning<GradientIterator, HessianIterator>::ZeroCallback zeroCallback) const override;

    };

}
