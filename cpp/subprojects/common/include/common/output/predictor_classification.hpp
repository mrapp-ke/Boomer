/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_matrix.hpp"
#include "common/input/label_matrix_row_wise.hpp"
#include "common/output/predictor_sparse.hpp"
#include "common/output/label_vector_set.hpp"
#include "common/model/rule_list.hpp"


/**
 * Defines an interface for all predictors that predict whether individual labels of given query examples are relevant
 * or irrelevant using an existing rule-based model.
 */
class IClassificationPredictor : public ISparsePredictor<uint8> {

    public:

        virtual ~IClassificationPredictor() override { };

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IClassificationPredictor`.
 */
class IClassificationPredictorFactory {

    public:

        virtual ~IClassificationPredictorFactory() { };

        /**
         * Creates and returns a new object of the type `IClassificationPredictor`.
         *
         * @param model             A reference to an object of type `RuleList` that should be used to obtain
         *                          predictions
         * @param labelVectorSet    A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         * @return                  An unique pointer to an object of type `IClassificationPredictor` that has been
         *                          created
         */
        virtual std::unique_ptr<IClassificationPredictor> create(const RuleList& model,
                                                                 const LabelVectorSet* labelVectorSet) const = 0;

};

/**
 * Defines an interface for all classes that allow to configure a predictor that predicts whether individual labels of
 * given query examples are relevant or irrelevant.
 */
class IClassificationPredictorConfig {

    public:

        virtual ~IClassificationPredictorConfig() { };

        /**
         * Creates and returns a new object of type `IClassificationPredictorFactory` according to the specified
         * configuration.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param numLabels     The total number of available labels
         * @return              An unique pointer to an object of type `IClassificationPredictorFactory` that has been
         *                      created
         */
        virtual std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new object of type `ILabelSpaceInfo` that is required by the predictor.
         *
         * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access to
         *                      the label of the training examples
         * @return              An unique pointer to an object of type `ILabelSpaceInfo` that has been created
         */
        virtual std::unique_ptr<ILabelSpaceInfo> createLabelSpaceInfo(const IRowWiseLabelMatrix& labelMatrix) const = 0;

};
