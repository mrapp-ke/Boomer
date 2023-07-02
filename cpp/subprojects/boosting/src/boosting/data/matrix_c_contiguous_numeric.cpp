#include "boosting/data/matrix_c_contiguous_numeric.hpp"

namespace boosting {

    template<typename T>
    NumericCContiguousMatrix<T>::NumericCContiguousMatrix(uint32 numRows, uint32 numCols)
        : CContiguousMatrix<T>(numRows, numCols) {}

    template<typename T>
    NumericCContiguousMatrix<T>::NumericCContiguousMatrix(uint32 numRows, uint32 numCols, bool init)
        : CContiguousMatrix<T>(numRows, numCols, init) {}

    template<typename T>
    void NumericCContiguousMatrix<T>::addToRowFromSubset(uint32 row, typename VectorConstView<T>::const_iterator begin,
                                                         typename VectorConstView<T>::const_iterator end,
                                                         CompleteIndexVector::const_iterator indicesBegin,
                                                         CompleteIndexVector::const_iterator indicesEnd) {
        typename NumericCContiguousMatrix<T>::value_iterator iterator = this->values_begin(row);
        uint32 numCols = this->getNumCols();

        for (uint32 i = 0; i < numCols; i++) {
            iterator[i] += begin[i];
        }
    }

    template<typename T>
    void NumericCContiguousMatrix<T>::addToRowFromSubset(uint32 row, typename VectorConstView<T>::const_iterator begin,
                                                         typename VectorConstView<T>::const_iterator end,
                                                         PartialIndexVector::const_iterator indicesBegin,
                                                         PartialIndexVector::const_iterator indicesEnd) {
        typename NumericCContiguousMatrix<T>::value_iterator iterator = this->values_begin(row);
        uint32 numCols = indicesEnd - indicesBegin;

        for (uint32 i = 0; i < numCols; i++) {
            uint32 index = indicesBegin[i];
            iterator[index] += begin[i];
        }
    }

    template<typename T>
    void NumericCContiguousMatrix<T>::removeFromRowFromSubset(uint32 row,
                                                              typename VectorConstView<T>::const_iterator begin,
                                                              typename VectorConstView<T>::const_iterator end,
                                                              CompleteIndexVector::const_iterator indicesBegin,
                                                              CompleteIndexVector::const_iterator indicesEnd) {
        typename NumericCContiguousMatrix<T>::value_iterator iterator = this->values_begin(row);
        uint32 numCols = this->getNumCols();

        for (uint32 i = 0; i < numCols; i++) {
            iterator[i] -= begin[i];
        }
    }

    template<typename T>
    void NumericCContiguousMatrix<T>::removeFromRowFromSubset(uint32 row,
                                                              typename VectorConstView<T>::const_iterator begin,
                                                              typename VectorConstView<T>::const_iterator end,
                                                              PartialIndexVector::const_iterator indicesBegin,
                                                              PartialIndexVector::const_iterator indicesEnd) {
        typename NumericCContiguousMatrix<T>::value_iterator iterator = this->values_begin(row);
        uint32 numCols = indicesEnd - indicesBegin;

        for (uint32 i = 0; i < numCols; i++) {
            uint32 index = indicesBegin[i];
            iterator[index] -= begin[i];
        }
    }

    template class NumericCContiguousMatrix<uint8>;
    template class NumericCContiguousMatrix<uint32>;
    template class NumericCContiguousMatrix<float32>;
    template class NumericCContiguousMatrix<float64>;

}
