#include "common/binning/bin_index_vector_dense.hpp"

#include "common/statistics/statistics_weighted.hpp"

DenseBinIndexVector::DenseBinIndexVector(uint32 numElements) : vector_(DenseVector<uint32>(numElements)) {}

uint32 DenseBinIndexVector::getBinIndex(uint32 exampleIndex) const {
    return vector_[exampleIndex];
}

void DenseBinIndexVector::setBinIndex(uint32 exampleIndex, uint32 binIndex) {
    vector_[exampleIndex] = binIndex;
}

std::unique_ptr<IHistogram> DenseBinIndexVector::createHistogram(const IWeightedStatistics& statistics,
                                                                 uint32 numBins) const {
    return statistics.createHistogram(*this, numBins);
}
