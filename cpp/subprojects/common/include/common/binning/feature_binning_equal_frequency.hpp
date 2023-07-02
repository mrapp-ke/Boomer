/*
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/binning/feature_binning.hpp"
#include "common/macros.hpp"
#include "common/multi_threading/multi_threading.hpp"

/**
 * Defines an interface for all classes that allow to configure a method that assigns numerical feature values to bins,
 * such that each bins contains approximately the same number of values.
 */
class MLRLCOMMON_API IEqualFrequencyFeatureBinningConfig {
    public:

        virtual ~IEqualFrequencyFeatureBinningConfig() {};

        /**
         * Returns the percentage that specifies how many bins are used.
         *
         * @return The percentage that specifies how many bins are used
         */
        virtual float32 getBinRatio() const = 0;

        /**
         * Sets the percentage that specifies how many bins should be used.
         *
         * @param binRatio  The percentage that specifies how many bins should be used, e.g., if 100 values are
         *                  available, a percentage of 0.5 means that `ceil(0.5 * 100) = 50` bins should be used. Must
         *                  be in (0, 1)
         * @return          A reference to an object of type `IEqualFrequencyFeatureBinningConfig` that allows further
         *                  configuration of the method that assigns numerical feature values to bins
         */
        virtual IEqualFrequencyFeatureBinningConfig& setBinRatio(float32 binRatio) = 0;

        /**
         * Returns the minimum number of bins that is used.
         *
         * @return The minimum number of bins that is used
         */
        virtual uint32 getMinBins() const = 0;

        /**
         * Sets the minimum number of bins that should be used.
         *
         * @param minBins   The minimum number of bins that should be used. Must be at least 2
         * @return          A reference to an object of type `IEqualFrequencyFeatureBinningConfig` that allows further
         *                  configuration of the method that assigns numerical feature values to bins
         */
        virtual IEqualFrequencyFeatureBinningConfig& setMinBins(uint32 minBins) = 0;

        /**
         * Returns the maximum number of bins that is used.
         *
         * @return The maximum number of bins that is used
         */
        virtual uint32 getMaxBins() const = 0;

        /**
         * Sets the maximum number of bins that should be used.
         *
         * @param maxBins   The maximum number of bins that should be used. Must be at least the minimum number of bins
         *                  or 0, if the maximum number of bins should not be restricted
         * @return          A reference to an object of type `IEqualFrequencyFeatureBinningConfig` that allows further
         *                  configuration of the method that assigns numerical feature values to bins
         */
        virtual IEqualFrequencyFeatureBinningConfig& setMaxBins(uint32 maxBins) = 0;
};

/**
 * Allows to configure a method that assigns numerical feature values to bins, such that each bins contains
 * approximately the same number of values.
 */
class EqualFrequencyFeatureBinningConfig final : public IFeatureBinningConfig,
                                                 public IEqualFrequencyFeatureBinningConfig {
    private:

        float32 binRatio_;

        uint32 minBins_;

        uint32 maxBins_;

        const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

    public:

        /**
         * @param multiThreadingConfigPtr A reference to an unique pointer that stores the configuration of the
         *                                multi-threading behavior that should be used for the parallel update of
         *                                statistics
         */
        EqualFrequencyFeatureBinningConfig(const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

        float32 getBinRatio() const override;

        IEqualFrequencyFeatureBinningConfig& setBinRatio(float32 binRatio) override;

        uint32 getMinBins() const override;

        IEqualFrequencyFeatureBinningConfig& setMinBins(uint32 minBins) override;

        uint32 getMaxBins() const override;

        IEqualFrequencyFeatureBinningConfig& setMaxBins(uint32 maxBins) override;

        std::unique_ptr<IThresholdsFactory> createThresholdsFactory(const IFeatureMatrix& featureMatrix,
                                                                    const ILabelMatrix& labelMatrix) const override;
};
