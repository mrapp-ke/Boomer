#include "common/data/vector_dense.hpp"
#include "common/data/indexed_value.hpp"
#include <cstdlib>


template<typename T>
DenseVector<T>::DenseVector(uint32 numElements)
    : DenseVector<T>(numElements, false) {

}

template<typename T>
DenseVector<T>::DenseVector(uint32 numElements, bool init)
    : VectorView<T>(numElements, (T*) (init ? calloc(numElements, sizeof(T)) : malloc(numElements * sizeof(T)))),
      maxCapacity_(numElements) {

}

template<typename T>
DenseVector<T>::~DenseVector() {
    free(this->array_);
}

template<typename T>
void DenseVector<T>::setNumElements(uint32 numElements, bool freeMemory) {
    if (numElements < maxCapacity_) {
        if (freeMemory) {
            this->array_ = (T*) realloc(this->array_, numElements * sizeof(T));
            maxCapacity_ = numElements;
        }
    } else if (numElements > maxCapacity_) {
        this->array_ = (T*) realloc(this->array_, numElements * sizeof(T));
        maxCapacity_ = numElements;
    }

    this->numElements_ = numElements;
}

template class DenseVector<uint8>;
template class DenseVector<uint32>;
template class DenseVector<float32>;
template class DenseVector<float64>;
template class DenseVector<IndexedValue<uint8>>;
template class DenseVector<IndexedValue<uint32>>;
template class DenseVector<IndexedValue<float32>>;
template class DenseVector<IndexedValue<float64>>;
