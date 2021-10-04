#include "seco/sampling/instance_sampling_with_replacement.hpp"
#include "seco/statistics/statistics.hpp"
#include "common/data/arrays.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/validation.hpp"
#include "instance_sampling_common.hpp"


namespace seco {

    static inline void sampleInternally(const uint32* exampleIndices, uint32 numExampleIndices, uint32 numSamples,
                                        DenseWeightVector<uint32>& weightVector, RNG& rng) {
        uint32 numExamples = weightVector.getNumElements();
        typename DenseWeightVector<uint32>::iterator weightIterator = weightVector.begin();
        setArrayToZeros(weightIterator, numExamples);
        uint32 numNonZeroWeights = 0;

        for (uint32 i = 0; i < numSamples; i++) {
            // Randomly select the index of an example...
            uint32 randomIndex = rng.random(0, numExampleIndices);
            uint32 exampleIndex = exampleIndices[randomIndex];

            // Update weight at the selected index...
            uint32 previousWeight = weightIterator[exampleIndex];
            weightIterator[exampleIndex] = previousWeight + 1;

            if (previousWeight == 0) {
                numNonZeroWeights++;
            }
        }

        weightVector.setNumNonZeroWeights(numNonZeroWeights);
    }

    /**
     * Allows to select a subset of the available training examples that have at least one label with non-zero weight
     * with replacement.
     *
     * @tparam Partition    The type of the object that provides access to the indices of the examples that are included
     *                      in the training set
     * @tparam WeightMatrix The type of the matrix that provides access to the weights of individual examples and labels
     */
    template<typename Partition, typename WeightMatrix>
    class InstanceSamplingWithReplacement final : public AbstractInstanceSampling<Partition, WeightMatrix> {

        private:

            float32 sampleSize_;

            DenseWeightVector<uint32> weightVector_;

        public:

            /**
             * @param partition     A reference to an object of template type `Partition` that provides access to the
             *                      indices of the examples that are included in the training set
             * @param weightMatrix  A reference to an object of template type `WeightMatrix` that provides access to the
             *                      weights of individual examples and labels
             * @param sampleSize    The fraction of examples to be included in the sample (e.g. a value of 0.6
             *                      corresponds to 60 % of the available examples). Must be in (0, 1)
             */
            InstanceSamplingWithReplacement(Partition& partition, const WeightMatrix& weightMatrix, float32 sampleSize)
                : AbstractInstanceSampling<Partition, WeightMatrix>(partition, weightMatrix), sampleSize_(sampleSize),
                  weightVector_(DenseWeightVector<uint32>(partition.getNumElements())) {

            }

        protected:

            const IWeightVector& sample(const uint32* exampleIndices, uint32 numExampleIndices, RNG& rng) override {
                uint32 numSamples = (uint32) (sampleSize_ * numExampleIndices);
                sampleInternally(exampleIndices, numExampleIndices, numSamples,  weightVector_, rng);
                return weightVector_;
            }

    };

    template<typename Partition>
    static inline std::unique_ptr<IInstanceSampling> createSampling(Partition& partition, IStatistics& statistics,
                                                                    float32 sampleSize) {
        std::unique_ptr<IInstanceSampling> instanceSamplingPtr;
        ICoverageStatistics::DenseWeightMatrixVisitor denseWeightMatrixVisitor =
            [&](std::unique_ptr<DenseWeightMatrix>& weightMatrixPtr) mutable {
                instanceSamplingPtr =
                    std::make_unique<InstanceSamplingWithReplacement<Partition, DenseWeightMatrix>>(partition,
                                                                                                    *weightMatrixPtr,
                                                                                                    sampleSize);
        };
        ICoverageStatistics& coverageStatistics = dynamic_cast<ICoverageStatistics&>(statistics);
        coverageStatistics.visitWeightMatrix(denseWeightMatrixVisitor);
        return instanceSamplingPtr;
    }

    InstanceSamplingWithReplacementFactory::InstanceSamplingWithReplacementFactory(float32 sampleSize)
        : sampleSize_(sampleSize) {
        assertGreater<float32>("sampleSize", sampleSize, 0);
        assertLessOrEqual<float32>("sampleSize", sampleSize, 1);
    }

    std::unique_ptr<IInstanceSampling> InstanceSamplingWithReplacementFactory::create(
            const CContiguousLabelMatrix& labelMatrix, const SinglePartition& partition,
            IStatistics& statistics) const {
        return createSampling<const SinglePartition>(partition, statistics, sampleSize_);
    }

    std::unique_ptr<IInstanceSampling> InstanceSamplingWithReplacementFactory::create(
            const CContiguousLabelMatrix& labelMatrix, BiPartition& partition, IStatistics& statistics) const {
        return createSampling<BiPartition>(partition, statistics, sampleSize_);
    }

    std::unique_ptr<IInstanceSampling> InstanceSamplingWithReplacementFactory::create(
            const CsrLabelMatrix& labelMatrix, const SinglePartition& partition, IStatistics& statistics) const {
        return createSampling<const SinglePartition>(partition, statistics, sampleSize_);
    }

    std::unique_ptr<IInstanceSampling> InstanceSamplingWithReplacementFactory::create(
            const CsrLabelMatrix& labelMatrix, BiPartition& partition, IStatistics& statistics) const {
        return createSampling<BiPartition>(partition, statistics, sampleSize_);
    }

}
