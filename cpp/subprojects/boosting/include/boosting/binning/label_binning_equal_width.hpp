/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/binning/label_binning.hpp"


namespace boosting {

    /**
     * Allows to create instances of the class `EqualWidthLabelBinning`.
     */
    class EqualWidthLabelBinningFactory final : public ILabelBinningFactory {

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
            EqualWidthLabelBinningFactory(float32 binRatio, uint32 minBins, uint32 maxBins);

            std::unique_ptr<ILabelBinning> create() const override;

    };

}
