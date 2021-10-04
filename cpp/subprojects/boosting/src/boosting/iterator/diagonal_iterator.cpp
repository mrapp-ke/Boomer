#include "boosting/iterator/diagonal_iterator.hpp"
#include "boosting/math/math.hpp"


namespace boosting {

    template<typename T>
    DiagonalConstIterator<T>::DiagonalConstIterator(const T* ptr, uint32 index)
        : ptr_(ptr), index_(index) {

    }

    template<typename T>
    typename DiagonalConstIterator<T>::reference DiagonalConstIterator<T>::operator[](uint32 index) const {
        return ptr_[triangularNumber(index + 1) - 1];
    }

    template<typename T>
    typename DiagonalConstIterator<T>::reference DiagonalConstIterator<T>::operator*() const {
        return ptr_[triangularNumber(index_ + 1) - 1];
    }

    template<typename T>
    DiagonalConstIterator<T>& DiagonalConstIterator<T>::operator++() {
        ++index_;
        return *this;
    }

    template<typename T>
    DiagonalConstIterator<T>& DiagonalConstIterator<T>::operator++(int n) {
        index_++;
        return *this;
    }

    template<typename T>
    DiagonalConstIterator<T>& DiagonalConstIterator<T>::operator--() {
        --index_;
        return *this;
    }

    template<typename T>
    DiagonalConstIterator<T>& DiagonalConstIterator<T>::operator--(int n) {
        index_--;
        return *this;
    }

    template<typename T>
    bool DiagonalConstIterator<T>::operator!=(const DiagonalConstIterator<T>& rhs) const {
        return index_ != rhs.index_;
    }

    template<typename T>
    bool DiagonalConstIterator<T>::operator==(const DiagonalConstIterator<T>& rhs) const {
        return index_ == rhs.index_;
    }

    template<typename T>
    typename DiagonalConstIterator<T>::difference_type DiagonalConstIterator<T>::operator-(
            const DiagonalConstIterator<T>& rhs) const {
        return (difference_type) index_ - (difference_type) rhs.index_;
    }

    template class DiagonalConstIterator<uint8>;
    template class DiagonalConstIterator<uint32>;
    template class DiagonalConstIterator<float32>;
    template class DiagonalConstIterator<float64>;

}
