/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"
#include "common/data/functions.hpp"
#include "common/input/label_matrix.hpp"


/**
 * Implements random read-only access to the labels of individual training examples that are stored in a pre-allocated
 * C-contiguous array.
 */
class CContiguousLabelMatrix final : public ILabelMatrix {

    private:

        CContiguousConstView<const uint8> view_;

    public:

        /**
         * Provides access to the values that are stored in a single row of a `CContiguousLabelMatrix`.
         */
        class View final : public VectorConstView<const uint8> {

            public:

                /**
                 * Allows to compute hash values for objects of type `CContiguousLabelMatrix::View`.
                 */
                struct Hash {

                    /**
                     * Computes and returns a hash value for an object of type `CContiguousLabelMatrix::View`.
                     *
                     * @param v A reference to an object of type `CContiguousLabelMatrix::View`
                     * @return  The hash value
                     */
                    inline std::size_t operator()(const View& v) const {
                        uint32 numElements = v.getNumElements();
                        std::size_t hashValue = (std::size_t) numElements;
                        View::const_iterator it = v.cbegin();

                        for (uint32 i = 0; i < numElements; i++) {
                            if (it[i]) {
                                hashValue ^= i + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
                            }
                        }

                        return hashValue;
                    }

                };

                /**
                 * Allows to check whether two objects of type `CContiguousLabelMatrix::View` are equal or not.
                 */
                struct Pred {

                    /**
                     * Returns whether two objects of type `CContiguousLabelMatrix::View` are equal or not.
                     *
                     * @param lhs   A reference to a first object of type `CContiguousLabelMatrix::View`
                     * @param rhs   A reference to a second object of type `CContiguousLabelMatrix::View`
                     * @return      True, if the given objects are equal, false otherwise
                     */
                    inline bool operator()(const View& lhs, const View& rhs) const {
                        return compareArrays(lhs.cbegin(), lhs.getNumElements(), rhs.cbegin(), rhs.getNumElements());
                    }

                };

                /**
                 * @param labelMatrix   A reference to an object of type `CContiguousLabelMatrix`, the view provides
                 *                      access to
                 * @param row           The row, the view provides access to
                 */
                View(const CContiguousLabelMatrix& labelMatrix, uint32 row);

        };

        /**
         * @param numRows   The number of rows in the label matrix
         * @param numCols   The number of columns in the label matrix
         * @param array     A pointer to a C-contiguous array of type `uint8` that stores the labels
         */
        CContiguousLabelMatrix(uint32 numRows, uint32 numCols, const uint8* array);

        /**
         * The type of the view that provides access to the values that are stored in a single row of the label matrix.
         */
        typedef const View view_type;

        /**
         * An iterator that provides read-only access to the values in the label matrix.
         */
        typedef CContiguousConstView<const uint8>::const_iterator value_const_iterator;

        /**
         * Returns a `value_const_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `value_const_iterator` to the beginning of the given row
         */
        value_const_iterator row_values_cbegin(uint32 row) const;

        /**
         * Returns a `value_const_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `value_const_iterator` to the end of the given row
         */
        value_const_iterator row_values_cend(uint32 row) const;

        /**
         * Creates and returns a view that provides access to the values at a specific row of the label matrix.
         *
         * @param row   The row
         * @return      An object of type `view_type` that has been created
         */
        view_type createView(uint32 row) const;

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

        std::unique_ptr<LabelVector> createLabelVector(uint32 row) const override;

        std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
            const IStatisticsProviderFactory& factory) const override;

        std::unique_ptr<IPartitionSampling> createPartitionSampling(
            const IPartitionSamplingFactory& factory) const override;

        std::unique_ptr<IInstanceSampling> createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                  const SinglePartition& partition,
                                                                  IStatistics& statistics) const override;

        std::unique_ptr<IInstanceSampling> createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                  BiPartition& partition,
                                                                  IStatistics& statistics) const override;

};
