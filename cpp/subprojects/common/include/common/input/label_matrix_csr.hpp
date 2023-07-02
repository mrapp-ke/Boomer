/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "common/data/arrays.hpp"
#include "common/data/view_csr_binary.hpp"
#include "common/data/view_vector.hpp"
#include "common/input/label_matrix_row_wise.hpp"

/**
 * Defines an interface for all label matrices that provide row-wise access to the labels of individual examples that
 * are stored in a sparse matrix in the compressed sparse row (CSR) format.
 */
class MLRLCOMMON_API ICsrLabelMatrix : virtual public IRowWiseLabelMatrix {
    public:

        virtual ~ICsrLabelMatrix() override {};
};

/**
 * Implements row-wise read-only access to the labels of individual training examples that are stored in a pre-allocated
 * sparse matrix in the compressed sparse row (CSR) format.
 */
class CsrLabelMatrix final : public BinaryCsrConstView,
                             virtual public ICsrLabelMatrix {
    public:

        /**
         * Provides access to the values that are stored in a single row of a `CsrLabelMatrix``.
         */
        class View final : public VectorConstView<const uint32> {
            public:

                /**
                 * Allows to compute hash values for objects of type `CsrLabelMatrix::View`.
                 */
                struct Hash final {
                    public:

                        /**
                         * Computes and returns a hash value for a given object of type `CsrLabelMatrix::View`.
                         *
                         * @param v A reference to an object of type `CsrLabelMatrix::View`
                         * @return  The hash value
                         */
                        inline std::size_t operator()(const View& v) const {
                            return hashArray(v.cbegin(), v.getNumElements());
                        }
                };

                /**
                 * Allows to check whether two objects of type `CsrLabelMatrix::View` are equal or not.
                 */
                struct Pred final {
                    public:

                        /**
                         * Returns whether two objects of type `CsrLabelMatrix::View` are equal or not.
                         *
                         * @param lhs   A reference to a first object of type `CsrLabelMatrix::View`
                         * @param rhs   A reference to a second object of type `CsrLabelMatrix::View`
                         * @return      True, if the given objects are equal, false otherwise
                         */
                        inline bool operator()(const View& lhs, const View& rhs) const {
                            return compareArrays(lhs.cbegin(), lhs.getNumElements(), rhs.cbegin(),
                                                 rhs.getNumElements());
                        }
                };

                /**
                 * @param labelMatrix   A reference to an object of type `CsrLabelMatrix`, the view provides access to
                 * @param row           The row, the view provides access to
                 */
                View(const CsrLabelMatrix& labelMatrix, uint32 row);
        };

        /**
         * @param numRows       The number of rows in the label matrix
         * @param numCols       The number of columns in the label matrix
         * @param rowIndices    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices
         *                      of the first element in `colIndices` that corresponds to a certain row. The index at the
         *                      last position is equal to `num_non_zero_values`
         * @param colIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
         *                      column-indices, the relevant labels correspond to
         */
        CsrLabelMatrix(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices);

        /**
         * The type of the view that provides access to the values that are stored in a single row of the label matrix.
         */
        typedef const View view_type;

        /**
         * Creates and returns a view that provides access to the values at a specific row of the label matrix.
         *
         * @param row   The row
         * @return      An object of type `view_type` that has been created
         */
        view_type createView(uint32 row) const;

        bool isSparse() const override;

        float32 calculateLabelCardinality() const override;

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

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
          const IMarginalProbabilityCalibrator& probabilityCalibrator, const SinglePartition& partition,
          const IStatistics& statistics) const override;

        std::unique_ptr<IMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
          const IMarginalProbabilityCalibrator& probabilityCalibrator, BiPartition& partition,
          const IStatistics& statistics) const override;

        std::unique_ptr<IJointProbabilityCalibrationModel> fitJointProbabilityCalibrationModel(
          const IJointProbabilityCalibrator& probabilityCalibrator, const SinglePartition& partition,
          const IStatistics& statistics) const override;

        std::unique_ptr<IJointProbabilityCalibrationModel> fitJointProbabilityCalibrationModel(
          const IJointProbabilityCalibrator& probabilityCalibrator, BiPartition& partition,
          const IStatistics& statistics) const override;
};

/**
 * Creates and returns a new object of the type `ICsrLabelMatrix`.
 *
 * @param numRows       The number of rows in the label matrix
 * @param numCols       The number of columns in the label matrix
 * @param rowIndices    A pointer to an array of type `uint32`, shape `(numRows + 1)`, that stores the indices
 *                      of the first element in `colIndices` that corresponds to a certain row. The index at the last
 *                      position is equal to `num_non_zero_values`
 * @param colIndices    A pointer to an array of type `uint32`, shape `(num_non_zero_values)`, that stores the
 *                      column-indices, the relevant labels correspond to
 * @return              An unique pointer to an object of type `ICsrLabelMatrix` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ICsrLabelMatrix> createCsrLabelMatrix(uint32 numRows, uint32 numCols, uint32* rowIndices,
                                                                     uint32* colIndices);

#ifdef _WIN32
    #pragma warning(pop)
#endif
