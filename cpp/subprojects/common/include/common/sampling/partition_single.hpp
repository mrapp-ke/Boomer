/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/partition.hpp"
#include "common/indices/index_iterator.hpp"


/**
 * An implementation of the class `IPartition` that provides random access to the indices of elements that are included
 * in a single set.
 */
class SinglePartition : public IPartition {

    private:

        uint32 numElements_;

    public:

        /**
         * @param numElements The number of elements to be included in the partition. Must be at least 1
         */
        SinglePartition(uint32 numElements);

        /**
         * An iterator that provides read-only access to the indices of the elements that are included in the partition.
         */
        typedef IndexIterator const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the indices that are contained in the partition.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the indices that are contained in the partition.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Returns the number of elements that are contained in the partition.
         *
         * @return The number of elements
         */
        uint32 getNumElements() const;

        std::unique_ptr<IWeightVector> subSample(const IInstanceSubSampling& instanceSubSampling,
                                                 RNG& rng) const override;

        float64 evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const ICoverageState& coverageState,
                                    const AbstractPrediction& head) override;

        void recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const ICoverageState& coverageState,
                                   Refinement& refinement) override;

};
