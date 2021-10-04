/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/instance_sampling.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "seco/data/matrix_weight_dense.hpp"


namespace seco {

    static inline uint32 updateExampleIndices(const SinglePartition& partition, const DenseWeightMatrix& weightMatrix,
                                              uint32* exampleIndices) {
        uint32 numTrainingExamples = partition.getNumElements();
        uint32 numLabels = weightMatrix.getNumCols();
        uint32 n = 0;

        for (uint32 i = 0; i < numTrainingExamples; i++) {
            DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(i);

            for (uint32 j = 0; j < numLabels; j++) {
                if (weightIterator[j] > 0) {
                    exampleIndices[n] = i;
                    n++;
                    break;
                }
            }
        }

        return n;
    }

    static inline uint32 updateExampleIndices(BiPartition& partition, const DenseWeightMatrix& weightMatrix,
                                              uint32* exampleIndices) {
        uint32 numTrainingExamples = partition.getNumFirst();
        uint32 numLabels = weightMatrix.getNumCols();
        uint32 n = 0;
        BiPartition::const_iterator indexIterator = partition.first_cbegin();

        for (uint32 i = 0; i < numTrainingExamples; i++) {
            uint32 index = indexIterator[i];
            DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(index);

            for (uint32 j = 0; j < numLabels; j++) {
                if (weightIterator[j] > 0) {
                    exampleIndices[n] = index;
                    n++;
                    break;
                }
            }
        }

        return n;
    }

    /**
     * An abstract base class for all classes that allow to select a subset of the available training examples that have
     * at least one label with non-zero weight.
     *
     * @tparam Partition    The type of the object that provides access to the indices of the examples that are included
     *                      in the training set
     * @tparam WeightMatrix The type of the matrix that provides access to the weights of individual examples and labels
     */
    template<typename Partition, typename WeightMatrix>
    class AbstractInstanceSampling : public IInstanceSampling {

        private:

            Partition& partition_;

            const WeightMatrix& weightMatrix_;

            uint32* exampleIndices_;

        protected:

            /**
             * Must be implemented by subclasses in order to select a subset of the training examples that have at least
             * one label with non-zero weight.
             *
             * @param exampleIndices    An array of type `uint32` that stores the indices of the training examples that
             *                          have at least one label with non-zero weight
             * @param numExampleIndices The number of elements in the array `exampleIndices`
             * @param rng               A reference to an object of type `RNG`, implementing the random number generator
             *                          to be used
             */
            virtual const IWeightVector& sample(const uint32* exampleIndices, uint32 numExampleIndices, RNG& rng) = 0;

        public:

            /**
             * @param partition     A reference to an object of template type `Partition` that provides access to the
             *                      indices of the examples that are included in the training set
             * @param weightMatrix  A reference to an object of template type `WeightMatrix` that provides access to the
             *                      weights of individual examples and labels
             */
            AbstractInstanceSampling(Partition& partition, const WeightMatrix& weightMatrix)
                : partition_(partition), weightMatrix_(weightMatrix),
                  exampleIndices_(new uint32[partition.getNumElements()]) {

            }

            virtual ~AbstractInstanceSampling() {
                delete[] exampleIndices_;
            }

            const IWeightVector& sample(RNG& rng) override final {
                uint32 numExampleIndices = updateExampleIndices(partition_, weightMatrix_, exampleIndices_);
                return this->sample(exampleIndices_, numExampleIndices, rng);
            }

    };

}
