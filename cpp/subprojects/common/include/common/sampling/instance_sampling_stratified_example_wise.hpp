/*
 * @author Anna Kulischkin (Anna_Kulischkin@web.de)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/instance_sampling.hpp"


/**
 * Allows to create instances of the type `IInstanceSampling` that implement stratified sampling, where distinct label
 * vectors are treated as individual classes.
 */
class ExampleWiseStratifiedSamplingFactory final : public IInstanceSamplingFactory {

    private:

        float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1]
         */
        ExampleWiseStratifiedSamplingFactory(float32 sampleSize);

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
