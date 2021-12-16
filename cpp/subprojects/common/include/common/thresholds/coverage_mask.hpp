/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/thresholds/coverage_state.hpp"


/**
 * Allows to check whether individual examples are covered by a rule or not. For each example, an integer is stored in a
 * C-contiguous array that may be updated when the rule is refined. If the value that corresponds to a certain example
 * is equal to the "indicator value", it is considered to be covered.
 */
class CoverageMask final : public ICoverageState {

    private:

        uint32* array_;

        uint32 numElements_;

        uint32 indicatorValue_;

    public:

        /**
         * @param numElements The total number of examples
         */
        CoverageMask(uint32 numElements);

        /**
         * @param coverageMask A reference to an object of type `CoverageMask` to be copied
         */
        CoverageMask(const CoverageMask& coverageMask);

        ~CoverageMask();

        /**
         * An iterator that provides access to the values in the mask and allows to modify them.
         */
        typedef uint32* iterator;

        /**
         * An iterator that provides read-only access to the values in the mask.
         */
        typedef const uint32* const_iterator;

        /**
         * Returns an `iterator` to the beginning of the mask.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the mask.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Returns a `const_iterator` to the beginning of the mask.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the mask.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Returns the total number of examples
         *
         * @return The total number of examples
         */
        uint32 getNumElements() const;

        /**
         * Returns the "indicator value".
         *
         * @return The "indicator value"
         */
        uint32 getIndicatorValue() const;

        /**
         * Sets the "indicator value".
         *
         * @param indicatorValue The "indicator value" to be set
         */
        void setIndicatorValue(uint32 indicatorValue);

        /**
         * Resets the mask and the "indicator value" such that all examples are marked as covered.
         */
        void reset();

        /**
         * Returns whether the example at a specific index is covered or not.
         *
         * @param pos   The index of the example
         * @return      True, if the example at the given index is covered, false otherwise
         */
        bool isCovered(uint32 pos) const;

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
