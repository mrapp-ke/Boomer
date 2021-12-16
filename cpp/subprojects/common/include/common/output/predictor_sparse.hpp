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

        virtual ~ISparsePredictor() { };

        /**
         * Obtains and returns sparse predictions for all examples in a C-contiguous matrix, using a specific rule-based
         * model.
         *
         * @param featureMatrix     A reference to an object of type `CContiguousFeatureMatrix` that stores the feature
         *                          values of the examples
         * @param numLabels         The number of labels to predict for
         * @param model             A reference to an object of type `RuleModel` that should be used to obtain the
         *                          predictions
         * @param labelVectors      A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         * @return                  An unique pointer to an object of type `BinarySparsePredictionMatrix` that stores
         *                          the predictions
         */
        virtual std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
            const CContiguousFeatureMatrix& featureMatrix, uint32 numLabels, const RuleModel& model,
            const LabelVectorSet* labelVectors) const = 0;

        /**
         * Obtains and returns sparse predictions for all examples in a sparse CSR matrix, using a specific rule-based
         * model.
         *
         * @param featureMatrix     A reference to an object of type `CsrFeatureMatrix` that stores the feature values
         *                          of the examples
         * @param numLabels         The number of labels to predict for
         * @param model             A reference to an object of type `RuleModel` that should be used to obtain the
         *                          predictions
         * @param labelVectors      A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         * @return                  An unique pointer to an object of type `BinarySparsePredictionMatrix` that stores
         *                          predictions
         */
        virtual std::unique_ptr<BinarySparsePredictionMatrix> predictSparse(
            const CsrFeatureMatrix& featureMatrix, uint32 numLabels, const RuleModel& model,
            const LabelVectorSet* labelVectors) const = 0;

};
