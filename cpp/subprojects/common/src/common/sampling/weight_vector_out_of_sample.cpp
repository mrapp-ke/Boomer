#include "common/sampling/weight_vector_out_of_sample.hpp"

#include "common/sampling/weight_vector_bit.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/weight_vector_equal.hpp"

template<typename WeightVector>
OutOfSampleWeightVector<WeightVector>::OutOfSampleWeightVector(const WeightVector& vector) : vector_(vector) {}

template<typename WeightVector>
uint32 OutOfSampleWeightVector<WeightVector>::getNumElements() const {
    return vector_.getNumElements();
}

template<typename WeightVector>
bool OutOfSampleWeightVector<WeightVector>::operator[](uint32 pos) const {
    return vector_[pos] == 0;
}

template class OutOfSampleWeightVector<EqualWeightVector>;
template class OutOfSampleWeightVector<BitWeightVector>;
template class OutOfSampleWeightVector<DenseWeightVector<uint32>>;
