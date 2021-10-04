#include "common/binning/bin_index_vector_dok.hpp"


DokBinIndexVector::DokBinIndexVector()
    : vector_(DokVector<uint32>(BIN_INDEX_SPARSE)) {

}

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
