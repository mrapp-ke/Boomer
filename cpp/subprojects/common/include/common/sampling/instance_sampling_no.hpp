/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/sampling/instance_sampling.hpp"


/**
 * Allows to create instances of the type `IInstanceSampling` that do not perform any sampling, but assign equal weights
 * to all examples.
 */
class NoInstanceSamplingFactory final : public IInstanceSamplingFactory {

    public:

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
