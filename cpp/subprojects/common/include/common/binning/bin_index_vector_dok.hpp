/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/binning/bin_index_vector.hpp"
#include "common/data/vector_dok.hpp"


/**
 * Stores the indices of the bins, individual examples have been assigned to, using the dictionaries of keys (DOK)
 * format.
 */
class DokBinIndexVector final : public IBinIndexVector {

    private:

        DokVector<uint32> vector_;

    public:

        DokBinIndexVector();

        /**
         * An iterator that provides access to the elements in the vector.
         */
        typedef DokVector<uint32>::iterator iterator;

        /**
         * Returns an `iterator` to the beginning of the vector.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        uint32 getBinIndex(uint32 exampleIndex) const override;

        void setBinIndex(uint32 exampleIndex, uint32 binIndex) override;

};
