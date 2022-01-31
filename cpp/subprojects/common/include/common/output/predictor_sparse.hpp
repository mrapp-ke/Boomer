/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/prediction_matrix_sparse_binary.hpp"
#include "common/output/predictor.hpp"


/**
 * Defines an interface for all classes that allow to make predictions for given query examples using an existing
 * rule-based model and write them into a sparse matrix.
 *
 * @tparam T The type of the values that are stored by the prediction matrix
 */
template<typename T>
class ISparsePredictor : public IPredictor<T> {

    public:

        virtual ~ISparsePredictor() override { };

        /**
         * Obtains and returns sparse predictions for all examples in a C-contiguous matrix, using a specific rule-based
         * model.
         *
         * @param featureMatrix A reference to an object of type `CContiguousConstView` that stores the feature values
         *                      of the examples
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `BinarySparsePredictionMatrix` that stores the
         *                      predictions
         */
        virtual std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
            const CContiguousConstView<const float32>& featureMatrix, uint32 numLabels) const = 0;

        /**
         * Obtains and returns sparse predictions for all examples in a sparse CSR matrix, using a specific rule-based
         * model.
         *
         * @param featureMatrix A reference to an object of type `CsrConstView` that stores the feature values of the
         *                      examples
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `BinarySparsePredictionMatrix` that stores
         *                      predictions
         */
        virtual std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
            const CsrConstView<const float32>& featureMatrix, uint32 numLabels) const = 0;

};
