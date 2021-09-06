#include "common/data/ring_buffer.hpp"


template<typename T>
RingBuffer<T>::RingBuffer(uint32 capacity)
    : array_(new T[capacity]), capacity_(capacity), pos_(0), full_(false) {

}

template<typename T>
RingBuffer<T>::~RingBuffer() {
    delete[] array_;
}

template<typename T>
typename RingBuffer<T>::const_iterator RingBuffer<T>::cbegin() const {
    return array_;
}

template<typename T>
typename RingBuffer<T>::const_iterator RingBuffer<T>::cend() const {
    return &array_[full_ ? capacity_ : pos_];
}

template<typename T>
uint32 RingBuffer<T>::getCapacity() const {
    return capacity_;
}

template<typename T>
uint32 RingBuffer<T>::getNumElements() const {
    return full_ ? capacity_ : pos_;
}

template<typename T>
bool RingBuffer<T>::isFull() const {
    return full_;
}

template<typename T>
std::pair<bool, T> RingBuffer<T>::push(T value) {
    std::pair<bool, T> result;
    result.first = full_;
    result.second = array_[pos_];
    array_[pos_] = value;
    pos_++;

    if (pos_ >= capacity_) {
        pos_ = 0;
        full_ = true;
    }

    return result;
}

template class RingBuffer<uint8>;
template class RingBuffer<uint32>;
template class RingBuffer<float32>;
template class RingBuffer<float64>;
