/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_matrix.hpp"
#include "common/output/predictor.hpp"
#include "common/output/label_vector_set.hpp"
#include "common/model/rule_list.hpp"


/**
 * Defines an interface for all predictors that predict label-wise regression scores for given query examples using an
 * existing rule-based model.
 */
class IRegressionPredictor : public IPredictor<float64> {

    public:

        virtual ~IRegressionPredictor() override { };

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IRegressionPredictor`.
 */
class IRegressionPredictorFactory {

    public:

        virtual ~IRegressionPredictorFactory() { };

        /**
         * Creates and returns a new object of the type `IRegressionPredictor`.
         *
         * @param model             A reference to an object of type `RuleList` that should be used to obtain
         *                          predictions
         * @param labelVectorSet    A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         * @return                  An unique pointer to an object of type `IRegressionPredictor` that has been created
         */
        virtual std::unique_ptr<IRegressionPredictor> create(const RuleList& model,
                                                             const LabelVectorSet* labelVectorSet) const = 0;

};


/**
 * Defines an interface for all classes that allow to configure a predictor that predicts label-wise regression scores
 * for given query examples.
 */
class IRegressionPredictorConfig {

    public:

        virtual ~IRegressionPredictorConfig() { };

        /**
         * Creates and returns a new object of type `IRegressionPredictorFactory` according to the specified
         * configuration.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param numLabels     The total number of available labels
         * @return              An unique pointer to an object of type `IRegressionPredictorFactory` that has been
         *                      created or a null pointer, if the prediction of regression scores is not supported
         */
        virtual std::unique_ptr<IRegressionPredictorFactory> createRegressionPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const = 0;

};
