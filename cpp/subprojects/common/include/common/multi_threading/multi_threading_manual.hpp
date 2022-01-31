/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/multi_threading/multi_threading.hpp"
#include "common/macros.hpp"


/**
 * Defines an interface for all classes that allow to configure the multi-threading behavior of a parallelizable
 * algorithm by manually specifying the number of threads to be used.
 */
class MLRLCOMMON_API IManualMultiThreadingConfig {

    public:

        virtual ~IManualMultiThreadingConfig() { };

        /**
         * Returns the number of threads that are used.
         *
         * @return The number of threads that are used or 0, if all available CPU cores are utilized
         */
        virtual uint32 getNumThreads() const = 0;

        /**
         * Sets the number of threads that should be used.
         *
         * @param numThreads    The number of threads that should be used. Must be at least 1 or 0, if all available CPU
         *                      cores should be utilized
         * @return              A reference to an object of type `IManualMultiThreadingConfig` that allows further
         *                      configuration of the multi-threading behavior
         */
        virtual IManualMultiThreadingConfig& setNumThreads(uint32 numThreads) = 0;

};

/**
 * Allows to configure the multi-threading behavior of a parallelizable algorithm by manually specifying the number of
 * threads to be used.
 */
class ManualMultiThreadingConfig final : public IMultiThreadingConfig, public IManualMultiThreadingConfig{

    private:

        uint32 numThreads_;

    public:

        ManualMultiThreadingConfig();

        uint32 getNumThreads() const override;

        IManualMultiThreadingConfig& setNumThreads(uint32 numThreads) override;

        uint32 getNumThreads(const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;

};
