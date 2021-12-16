/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/binning/bin_index_vector.hpp"
#include "common/data/vector_dense.hpp"


/**
 * Stores the indices of the bins, individual examples have been assigned to, using a C-contiguous array.
 */
class DenseBinIndexVector final : public IBinIndexVector {

    private:

        DenseVector<uint32> vector_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        DenseBinIndexVector(uint32 numElements);

        uint32 getBinIndex(uint32 exampleIndex) const override;

        void setBinIndex(uint32 exampleIndex, uint32 binIndex) override;

};
