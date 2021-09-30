/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_fortran_contiguous.hpp"
#include "common/input/feature_matrix.hpp"


/**
 * Implements column-wise read-only access to the feature values of individual training examples that are stored in a
 * pre-allocated Fortran-contiguous array.
 */
class FortranContiguousFeatureMatrix final : public IFeatureMatrix {

    private:

        FortranContiguousConstView<const float32> view_;

    public:

        /**
         * @param numRows   The number of rows in the feature matrix
         * @param numCols   The number of columns in the feature matrix
         * @param array     A pointer to a Fortran-contiguous array of type `float32` that stores the feature values
         */
        FortranContiguousFeatureMatrix(uint32 numRows, uint32 numCols, const float32* array);

        /**
         * An iterator that provides read-only access to the feature values.
         */
        typedef FortranContiguousConstView<const float32>::const_iterator const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of a specific column.
         *
         * @param col   The column
         * @return      A `const_iterator` to the beginning
         */
        const_iterator column_cbegin(uint32 col) const;

        /**
         * Returns a `const_iterator` to the end of a specific column.
         *
         * @param col   The column
         * @return      A `const_iterator` to the end
         */
        const_iterator column_cend(uint32 col) const;

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        void fetchFeatureVector(uint32 featureIndex, std::unique_ptr<FeatureVector>& featureVectorPtr) const override;

};
