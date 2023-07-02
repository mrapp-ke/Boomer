#include "common/binning/bin_index_vector_dok.hpp"

#include "common/statistics/statistics_weighted.hpp"

DokBinIndexVector::DokBinIndexVector() : vector_(DokVector<uint32>(BIN_INDEX_SPARSE)) {}

DokBinIndexVector::iterator DokBinIndexVector::begin() {
    return vector_.begin();
}

DokBinIndexVector::iterator DokBinIndexVector::end() {
    return vector_.end();
}

uint32 DokBinIndexVector::getBinIndex(uint32 exampleIndex) const {
    return vector_[exampleIndex];
}

void DokBinIndexVector::setBinIndex(uint32 exampleIndex, uint32 binIndex) {
    vector_.set(exampleIndex, binIndex);
}

std::unique_ptr<IHistogram> DokBinIndexVector::createHistogram(const IWeightedStatistics& statistics,
                                                               uint32 numBins) const {
    return statistics.createHistogram(*this, numBins);
}
