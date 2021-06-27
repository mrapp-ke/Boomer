#include "common/binning/bin_index_vector_dense.hpp"


DenseBinIndexVector::DenseBinIndexVector(uint32 numElements)
    : vector_(DenseVector<uint32>(numElements)) {

}

uint32 DenseBinIndexVector::getBinIndex(uint32 exampleIndex) const {
    return vector_.getValue(exampleIndex);
}

void DenseBinIndexVector::setBinIndex(uint32 exampleIndex, uint32 binIndex) {
    vector_.setValue(exampleIndex, binIndex);
}
