#include "boosting/binning/label_binning_equal_width.hpp"
#include "common/binning/binning.hpp"
#include "common/validation.hpp"
#include <limits>


namespace boosting {

    /**
     * Assigns labels to bins, based on the corresponding gradients and Hessians, in a way such that each bin contains
     * labels for which the predicted score is expected to belong to the same value range.
     */
    class EqualWidthLabelBinning final : public ILabelBinning {

        private:

            float32 binRatio_;

            uint32 minBins_;

            uint32 maxBins_;

        public:

            /**
             * @param binRatio  A percentage that specifies how many bins should be used to assign labels to, e.g., if
             *                  100 labels are available, 0.5 means that `ceil(0.5 * 100) = 50` bins should be used.
             *                  Must be in (0, 1)
             * @param minBins   The minimum number of bins to be used to assign labels to. Must be at least 2
             * @param maxBins   The maximum number of bins to be used to assign labels to. Must be at least `minBins` or
             *                  0, if the maximum number of bins should not be restricted
             */
            EqualWidthLabelBinning(float32 binRatio, uint32 minBins, uint32 maxBins)
                : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {
                assertGreater<float32>("binRatio", binRatio, 0.0);
                assertLess<uint32>("binRatio", binRatio, 1.0);
                assertGreaterOrEqual<uint32>("minBins", minBins, 1);
                if (maxBins != 0) { assertGreaterOrEqual<uint32>("maxBins", maxBins, minBins); }
            }

            uint32 getMaxBins(uint32 numLabels) const override {
                return calculateNumBins(numLabels, binRatio_, minBins_, maxBins_) + 1;
            }

            LabelInfo getLabelInfo(const float64* criteria, uint32 numElements) const override {
                LabelInfo labelInfo;
                labelInfo.numNegativeBins = 0;
                labelInfo.numPositiveBins = 0;

                if (numElements > 0) {
                    labelInfo.minNegative = 0;
                    labelInfo.maxNegative = -std::numeric_limits<float64>::infinity();
                    labelInfo.minPositive = std::numeric_limits<float64>::infinity();
                    labelInfo.maxPositive = 0;

                    for (uint32 i = 0; i < numElements; i++) {
                        float64 criterion = criteria[i];

                        if (criterion < 0) {
                            labelInfo.numNegativeBins++;

                            if (criterion < labelInfo.minNegative) {
                                labelInfo.minNegative = criterion;
                            }

                            if (criterion > labelInfo.maxNegative) {
                                labelInfo.maxNegative = criterion;
                            }
                        } else if (criterion > 0) {
                            labelInfo.numPositiveBins++;

                            if (criterion < labelInfo.minPositive) {
                                labelInfo.minPositive = criterion;
                            }

                            if (criterion > labelInfo.maxPositive) {
                                labelInfo.maxPositive = criterion;
                            }
                        }
                    }

                    if (labelInfo.numNegativeBins > 0) {
                        labelInfo.numNegativeBins = calculateNumBins(labelInfo.numNegativeBins, binRatio_, minBins_,
                                                                     maxBins_);
                    }

                    if (labelInfo.numPositiveBins > 0) {
                        labelInfo.numPositiveBins = calculateNumBins(labelInfo.numPositiveBins, binRatio_, minBins_,
                                                                     maxBins_);
                    }
                }

                return labelInfo;
            }

            void createBins(LabelInfo labelInfo, const float64* criteria, uint32 numElements, Callback callback,
                            ZeroCallback zeroCallback) const override {
                uint32 numNegativeBins = labelInfo.numNegativeBins;
                float64 minNegative = labelInfo.minNegative;
                float64 maxNegative = labelInfo.maxNegative;
                uint32 numPositiveBins = labelInfo.numPositiveBins;
                float64 minPositive = labelInfo.minPositive;
                float64 maxPositive = labelInfo.maxPositive;

                float64 spanPerNegativeBin = minNegative < 0 ? (maxNegative - minNegative) / numNegativeBins : 0;
                float64 spanPerPositiveBin = maxPositive > 0 ? (maxPositive - minPositive) / numPositiveBins : 0;

                for (uint32 i = 0; i < numElements; i++) {
                    float64 criterion = criteria[i];

                    if (criterion < 0) {
                        uint32 binIndex = std::floor((criterion - minNegative) / spanPerNegativeBin);

                        if (binIndex >= numNegativeBins) {
                            binIndex = numNegativeBins - 1;
                        }

                        callback(binIndex, i);
                    } else if (criterion > 0) {
                        uint32 binIndex = std::floor((criterion - minPositive) / spanPerPositiveBin);

                        if (binIndex >= numPositiveBins) {
                            binIndex = numPositiveBins - 1;
                        }

                        callback(numNegativeBins + binIndex, i);
                    } else {
                        zeroCallback(i);
                    }
                }
            }

    };

    EqualWidthLabelBinningFactory::EqualWidthLabelBinningFactory(float32 binRatio, uint32 minBins, uint32 maxBins)
        : binRatio_(binRatio), minBins_(minBins), maxBins_(maxBins) {

    }

    std::unique_ptr<ILabelBinning> EqualWidthLabelBinningFactory::create() const {
        return std::make_unique<EqualWidthLabelBinning>(binRatio_, minBins_, maxBins_);
    }

}
