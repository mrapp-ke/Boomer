/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/vector_dense.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/sampling/partition.hpp"

/**
 * An implementation of the class `IPartition` that provides random access to the indices of elements that are included
 * two, mutually exclusive, sets.
 */
class BiPartition final : public IPartition {
    private:

        DenseVector<uint32> vector_;

        const uint32 numFirst_;

        bool firstSorted_;

        bool secondSorted_;

    public:

        /**
         * @param numFirst  The number of elements that are contained by the first set
         * @param numSecond The number of elements that are contained by the second set
         */
        BiPartition(uint32 numFirst, uint32 numSecond);

        /**
         * An iterator that provides access to the indices that are contained by the first or second set and allows to
         * modify them.
         */
        typedef DenseVector<uint32>::iterator iterator;

        /**
         * An iterator that provides read-only access to the indices that are contained in the first or second set.
         */
        typedef DenseVector<uint32>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of the elements that are contained by the first set.
         *
         * @return An `iterator` to the beginning of the first set
         */
        iterator first_begin();

        /**
         * Returns an `iterator` to the end of the elements that are contained by the first set.
         *
         * @return An `iterator` to the end of the first set
         */
        iterator first_end();

        /**
         * Returns a `const_iterator` to the beginning of the elements that are contained by the first set.
         *
         * @return A `const_iterator` to the beginning of the first set
         */
        const_iterator first_cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the elements that are contained by the first set.
         *
         * @return A `const_iterator` to the end of the first set
         */
        const_iterator first_cend() const;

        /**
         * Returns an `iterator` to the beginning of the elements that are contained by the second set.
         *
         * @return An `iterator` to the beginning of the second set
         */
        iterator second_begin();

        /**
         * Returns an `iterator` to the beginning of the elements that are contained by the second set.
         *
         * @return An `iterator` to the beginning of the second set
         */
        iterator second_end();

        /**
         * Returns a `const_iterator` to the beginning of the elements that are contained by the second set.
         *
         * @return A `const_iterator` to the beginning of the second set
         */
        const_iterator second_cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the elements that are contained by the second set.
         *
         * @return A `const_iterator` to the end of the second set
         */
        const_iterator second_cend() const;

        /**
         * Returns the number of elements that are contained by the first set.
         *
         * @return The number of elements that are contained by the first set
         */
        uint32 getNumFirst() const;

        /**
         * Returns the number of elements that are contained by the second set.
         *
         * @return The number of elements that are contained by the second set
         */
        uint32 getNumSecond() const;

        /**
         * Sorts the elements that are contained by the first set in increasing order.
         */
        void sortFirst();

        /**
         * Sorts the elements that are contained by the second set in increasing order.
         */
        void sortSecond();

        /**
         * Returns the total number of elements.
         *
         * @return The total number of elements
         */
        uint32 getNumElements() const;

        std::unique_ptr<IStoppingCriterion> createStoppingCriterion(const IStoppingCriterionFactory& factory) override;

        std::unique_ptr<IInstanceSampling> createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                  const IRowWiseLabelMatrix& labelMatrix,
                                                                  IStatistics& statistics) override;

        Quality evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const ICoverageState& coverageState,
                                    const AbstractPrediction& head) override;

        void recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const ICoverageState& coverageState,
                                   AbstractPrediction& head) override;

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
          const IMarginalProbabilityCalibrator& probabilityCalibrator, const IRowWiseLabelMatrix& labelMatrix,
          const IStatistics& statistics) override;

        std::unique_ptr<IJointProbabilityCalibrationModel> fitJointProbabilityCalibrationModel(
          const IJointProbabilityCalibrator& probabilityCalibrator, const IRowWiseLabelMatrix& labelMatrix,
          const IStatistics& statistics) override;
};
