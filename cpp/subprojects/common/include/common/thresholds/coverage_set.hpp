/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/thresholds/coverage_state.hpp"


/**
 * Provides access to the indices of the examples that are covered by a rule. The indices of the covered examples are
 * stored in a C-contiguous array that may be updated when the rule is refined.
 */
class CoverageSet final : public ICoverageState {

    private:

        uint32* array_;

        uint32 numElements_;

        uint32 numCovered_;

    public:

        /**
         * @param numElements The total number of examples
         */
        CoverageSet(uint32 numElements);

        /**
         * @param coverageSet A reference to an object of type `CoverageSet` to be copied
         */
        CoverageSet(const CoverageSet& coverageSet);

        ~CoverageSet();

        /**
         * An iterator that provides access to the indices of the covered examples and allows to modify them.
         */
        typedef uint32* iterator;

        /**
         * An iterator that provides read-only access to the indices of the covered examples.
         */
        typedef const uint32* const_iterator;

        /**
         * Returns an `iterator` to the beginning of the indices of the covered examples.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the indices of the covered examples.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Returns a `const_iterator` to the beginning of the indices of the covered examples.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the indices of the covered examples.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Returns the total number of examples.
         *
         * @return The total number of examples
         */
        uint32 getNumElements() const;

        /**
         * Returns the number of covered examples.
         *
         * @return The number of covered examples
         */
        uint32 getNumCovered() const;

        /**
         * Sets the number of covered examples.
         *
         * @param numCovered The number of covered examples to be set
         */
        void setNumCovered(uint32 numCovered);

        /**
         * Resets the number of covered examples and their indices such that all examples are marked as covered.
         */
        void reset();

        std::unique_ptr<ICoverageState> copy() const override;

        float64 evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                    const AbstractPrediction& head) const override;

        float64 evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                    const AbstractPrediction& head) const override;

        void recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const SinglePartition& partition,
                                   Refinement& refinement) const override;

        void recalculatePrediction(const IThresholdsSubset& thresholdsSubset, BiPartition& partition,
                                   Refinement& refinement) const override;

};
