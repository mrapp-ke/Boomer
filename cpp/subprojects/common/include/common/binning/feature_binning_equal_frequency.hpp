/*
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/binning/feature_binning.hpp"


/**
 * Assigns feature values to bins in a way such that each bin contains approximately the same number of values.
 */
class EqualFrequencyFeatureBinning final : public IFeatureBinning {

    private:

        float32 binRatio_;

        uint32 minBins_;

        uint32 maxBins_;

    public:

        /**
         * @param binRatio  A percentage that specifies how many bins should be used, e.g., if 100 values are available,
         *                  0.5 means that `ceil(0.5 * 100) = 50` bins should be used. Must be in (0, 1)
         * @param minBins   The minimum number of bins to be used. Must be at least 2
         * @param maxBins   The maximum number of bins to be used. Must be at least `minBins` or 0, if the maximum
         *                  number of bins should not be restricted
         */
        EqualFrequencyFeatureBinning(float32 binRatio, uint32 minBins, uint32 maxBins);

        Result createBins(FeatureVector& featureVector, uint32 numExamples) const override;

};
