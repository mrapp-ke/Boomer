/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/instance_sampling.hpp"


/**
 * Allows to create instances of the type `IInstanceSampling` that allow to select a subset of the available training
 * examples without replacement.
 */
class InstanceSamplingWithoutReplacementFactory final : public IInstanceSamplingFactory {

    private:

        float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1)
         */
        InstanceSamplingWithoutReplacementFactory(float32 sampleSize);

        std::unique_ptr<IInstanceSampling> create(const CContiguousLabelMatrix& labelMatrix,
                                                  const SinglePartition& partition,
                                                  IStatistics& statistics) const override;

        std::unique_ptr<IInstanceSampling> create(const CContiguousLabelMatrix& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics) const override;

        std::unique_ptr<IInstanceSampling> create(const CsrLabelMatrix& labelMatrix, const SinglePartition& partition,
                                                  IStatistics& statistics) const override;

        std::unique_ptr<IInstanceSampling> create(const CsrLabelMatrix& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics) const override;

};
