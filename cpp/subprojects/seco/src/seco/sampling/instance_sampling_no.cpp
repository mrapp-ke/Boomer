#include "seco/sampling/instance_sampling_no.hpp"
#include "seco/statistics/statistics.hpp"
#include "seco/data/matrix_weight_dense.hpp"
#include "common/sampling/weight_vector_bit.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"


namespace seco {

    static inline void sampleInternally(const SinglePartition& partition, const DenseWeightMatrix& weightMatrix,
                                        BitWeightVector& weightVector, RNG& rng) {
        uint32 numTrainingExamples = weightVector.getNumElements();
        uint32 numLabels = weightMatrix.getNumCols();
        uint32 numNonZeroWeights = 0;
        weightVector.clear();

        for (uint32 i = 0; i < numTrainingExamples; i++) {
            DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(i);

            for (uint32 j = 0; j < numLabels; j++) {
                if (weightIterator[j] > 0) {
                    weightVector.set(i, true);
                    numNonZeroWeights++;
                    break;
                }
            }
        }

        weightVector.setNumNonZeroWeights(numNonZeroWeights);
    }

    static inline void sampleInternally(BiPartition& partition, const DenseWeightMatrix& weightMatrix,
                                        BitWeightVector& weightVector, RNG& rng) {
        uint32 numTrainingExamples = partition.getNumFirst();
        uint32 numLabels = weightMatrix.getNumCols();
        uint32 numNonZeroWeights = 0;
        BiPartition::const_iterator indexIterator = partition.first_cbegin();
        weightVector.clear();

        for (uint32 i = 0; i < numTrainingExamples; i++) {
            uint32 index = indexIterator[i];
            DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(index);

            for (uint32 j = 0; j < numLabels; j++) {
                if (weightIterator[j] > 0) {
                    weightVector.set(index, true);
                    numNonZeroWeights++;
                    break;
                }
            }
        }

        weightVector.setNumNonZeroWeights(numNonZeroWeights);
    }

    /**
     * An implementation of the class `IInstanceSampling` that does not perform any sampling, but assigns equal weights to
     * all examples that have at least one label with non-zero weight.
     *
     * @tparam Partition    The type of the object that provides access to the indices of the examples that are included
     *                      in the training set
     * @tparam WeightMatrix The type of the matrix that provides access to the weights of individual examples and labels
     */
    template<typename Partition, typename WeightMatrix>
    class NoInstanceSampling final : public IInstanceSampling {

        private:

            Partition& partition_;

            const WeightMatrix& weightMatrix_;

            BitWeightVector weightVector_;

        public:

            /**
             * @param partition     A reference to an object of template type `Partition` that provides access to the
             *                      indices of the examples that are included in the training set
             * @param weightMatrix  A reference to an object of template type `WeightMatrix` that provides access to the
             *                      weights of individual examples and labels
             */
            NoInstanceSampling(Partition& partition, const WeightMatrix& weightMatrix)
                : partition_(partition), weightMatrix_(weightMatrix),
                  weightVector_(BitWeightVector(partition.getNumElements())) {

            }

            const IWeightVector& sample(RNG& rng) override {
                sampleInternally(partition_, weightMatrix_, weightVector_, rng);
                return weightVector_;
            }

    };

    template<typename Partition>
    static inline std::unique_ptr<IInstanceSampling> createSampling(Partition& partition, IStatistics& statistics) {
        std::unique_ptr<IInstanceSampling> instanceSamplingPtr;
        ICoverageStatistics::DenseWeightMatrixVisitor denseWeightMatrixVisitor =
            [&](std::unique_ptr<DenseWeightMatrix>& weightMatrixPtr) mutable {
                instanceSamplingPtr = std::make_unique<NoInstanceSampling<Partition, DenseWeightMatrix>>(
                    partition, *weightMatrixPtr);
        };
        ICoverageStatistics& coverageStatistics = dynamic_cast<ICoverageStatistics&>(statistics);
        coverageStatistics.visitWeightMatrix(denseWeightMatrixVisitor);
        return instanceSamplingPtr;
    }

    std::unique_ptr<IInstanceSampling> NoInstanceSamplingFactory::create(const CContiguousLabelMatrix& labelMatrix,
                                                                         const SinglePartition& partition,
                                                                         IStatistics& statistics) const {
        return createSampling<const SinglePartition>(partition, statistics);
    }

    std::unique_ptr<IInstanceSampling> NoInstanceSamplingFactory::create(const CContiguousLabelMatrix& labelMatrix,
                                                                         BiPartition& partition,
                                                                         IStatistics& statistics) const {
        return createSampling<BiPartition>(partition, statistics);
    }

    std::unique_ptr<IInstanceSampling> NoInstanceSamplingFactory::create(const CsrLabelMatrix& labelMatrix,
                                                                         const SinglePartition& partition,
                                                                         IStatistics& statistics) const {
        return createSampling<const SinglePartition>(partition, statistics);
    }

    std::unique_ptr<IInstanceSampling> NoInstanceSamplingFactory::create(const CsrLabelMatrix& labelMatrix,
                                                                         BiPartition& partition,
                                                                         IStatistics& statistics) const {
        return createSampling<BiPartition>(partition, statistics);
    }

}
