#include "common/data/vector_dok.hpp"

template<typename T>
DokVector<T>::DokVector(T sparseValue)
    : sparseValue_(sparseValue) {

}

template<typename T>
typename DokVector<T>::iterator DokVector<T>::begin() {
    return data_.begin();
}

template<typename T>
typename DokVector<T>::iterator DokVector<T>::end() {
    return data_.end();
}

template<typename T>
typename DokVector<T>::const_iterator DokVector<T>::cbegin() const {
    return data_.cbegin();
}

template<typename T>
typename DokVector<T>::const_iterator DokVector<T>::cend() const {
    return data_.cend();
}

template<typename T>
const T& DokVector<T>::operator[](uint32 pos) const {
    auto it = data_.find(pos);
    return it != data_.cend() ? it->second : sparseValue_;
}

template<typename T>
void DokVector<T>::set(uint32 pos, T value) {
    auto result = data_.emplace(pos, value);

    if (!result.second) {
        result.first->second = value;
    }
}

template<typename T>
void DokVector<T>::clear() {
    data_.clear();
}

template class DokVector<uint8>;
template class DokVector<uint32>;
template class DokVector<float32>;
template class DokVector<float64>;
