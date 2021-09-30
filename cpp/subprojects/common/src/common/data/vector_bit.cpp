#include "common/data/vector_bit.hpp"
#include "common/data/arrays.hpp"
#include <climits>


constexpr std::size_t UINT32_SIZE = CHAR_BIT * sizeof(uint32);

static inline constexpr std::size_t size(uint32 numElements) {
    return (numElements + UINT32_SIZE - 1) / UINT32_SIZE;
}

static inline constexpr uint32 index(uint32 pos) {
    return pos / UINT32_SIZE;
}

static inline constexpr uint32 mask(uint32 pos) {
    return 1U << (pos % UINT32_SIZE);
}

BitVector::BitVector(uint32 numElements)
    : BitVector(numElements, false) {

}

BitVector::BitVector(uint32 numElements, bool init)
    : numElements_(numElements), array_(init ? new uint32[size(numElements)]{} : new uint32[size(numElements)]) {

}

BitVector::~BitVector() {
    delete[] array_;
}

bool BitVector::operator[](uint32 pos) const {
    return array_[index(pos)] & mask(pos);
}

void BitVector::set(uint32 pos, bool value) {
    if (value) {
        array_[index(pos)] |= mask(pos);
    } else {
        array_[index(pos)] &= ~mask(pos);
    }
}

uint32 BitVector::getNumElements() const {
    return numElements_;
}

void BitVector::clear() {
    setArrayToZeros(array_, size(numElements_));
}
