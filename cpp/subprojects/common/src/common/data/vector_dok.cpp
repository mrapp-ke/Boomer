#include "common/data/vector_dok.hpp"

template<class T>
DokVector<T>::DokVector(T sparseValue)
    : sparseValue_(sparseValue) {

}

template<class T>
typename DokVector<T>::iterator DokVector<T>::begin() {
    return data_.begin();
}

template<class T>
typename DokVector<T>::iterator DokVector<T>::end() {
    return data_.end();
}

template<class T>
typename DokVector<T>::const_iterator DokVector<T>::cbegin() const {
    return data_.cbegin();
}

template<class T>
typename DokVector<T>::const_iterator DokVector<T>::cend() const {
    return data_.cend();
}

template<class T>
T DokVector<T>::getValue(uint32 pos) const {
    auto it = data_.find(pos);
    return it != data_.cend() ? it->second : sparseValue_;
}

template<class T>
void DokVector<T>::setValue(uint32 pos, T value) {
    auto result = data_.emplace(pos, value);

    if (!result.second) {
        result.first->second = value;
    }
}

template<class T>
void DokVector<T>::setAllToZero() {
    data_.clear();
}

template class DokVector<uint32>;
