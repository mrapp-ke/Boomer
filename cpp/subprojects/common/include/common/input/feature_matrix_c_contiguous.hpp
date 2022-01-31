/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning( push )
    #pragma warning( disable : 4250 )
#endif

#include "common/input/feature_matrix_row_wise.hpp"
#include "common/data/view_c_contiguous.hpp"


/**
 * Defines an interface for all feature matrices that provide row-wise access to the feature values of examples that are
 * stored in a C-contiguous array.
 */
class MLRLCOMMON_API ICContiguousFeatureMatrix : virtual public IRowWiseFeatureMatrix {

    public:

        virtual ~ICContiguousFeatureMatrix() override { };

};

/**
 * An implementation of the type `ICContiguousFeatureMatrix` that provides row-wise read-only access to the feature
 * values of examples that are stored in a C-contiguous array.
 */
class CContiguousFeatureMatrix final : public CContiguousConstView<const float32>,
                                       virtual public ICContiguousFeatureMatrix {

    public:

        /**
         * @param numRows   The number of rows in the feature matrix
         * @param numCols   The number of columns in the feature matrix
         * @param array     A pointer to a C-contiguous array of type `float32` that stores the values, the feature
         *                  matrix provides access to
         */
        CContiguousFeatureMatrix(uint32 numRows, uint32 numCols, const float32* array);

        bool isSparse() const override;

        std::unique_ptr<DensePredictionMatrix<uint8>> predictLabels(const IClassificationPredictor& predictor,
                                                                    uint32 numLabels) const override;

        std::unique_ptr<BinarySparsePredictionMatrix> predictSparseLabels(const IClassificationPredictor& predictor,
                                                                          uint32 numLabels) const override;

        std::unique_ptr<DensePredictionMatrix<float64>> predictScores(const IRegressionPredictor& predictor,
                                                                      uint32 numLabels) const override;

        std::unique_ptr<DensePredictionMatrix<float64>> predictProbabilities(const IProbabilityPredictor& predictor,
                                                                             uint32 numLabels) const override;

};

/**
 * Creates and returns a new object of the type `ICContiguousFeatureMatrix`.
 *
 * @param numRows   The number of rows in the feature matrix
 * @param numCols   The number of columns in the feature matrix
 * @param array     A pointer to a C-contiguous array of type `float32` that stores the values, the feature matrix
 *                  provides access to
 * @return          An unique pointer to an object of type `ICContiguousFeatureMatrix` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ICContiguousFeatureMatrix> createCContiguousFeatureMatrix(uint32 numRows, uint32 numCols,
                                                                                         const float32* array);

#ifdef _WIN32
    #pragma warning ( pop )
#endif
