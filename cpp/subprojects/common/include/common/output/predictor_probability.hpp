/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_matrix.hpp"
#include "common/output/predictor.hpp"
#include "common/output/label_vector_set.hpp"
#include "common/model/rule_list.hpp"


/**
 * Defines an interface for all predictors that predict label-wise probabilities for given query examples, estimating
 * the chance of individual labels to be relevant, using an existing rule-based model.
 */
class IProbabilityPredictor : public IPredictor<float64> {

    public:

        virtual ~IProbabilityPredictor() override { };

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IProbabilityPredictor`.
 */
class IProbabilityPredictorFactory {

    public:

        virtual ~IProbabilityPredictorFactory() { };

        /**
         * Creates and returns a new object of the type `IProbabilityPredictor`.
         *
         * @param model             A reference to an object of type `RuleList` that should be used to obtain
         *                          predictions
         * @param labelVectorSet    A pointer to an object of type `LabelVectorSet` that stores all known label vectors
         *                          or a null pointer, if no such set is available
         * @return                  An unique pointer to an object of type `IProbabilityPredictor` that has been created
         */
        virtual std::unique_ptr<IProbabilityPredictor> create(const RuleList& model,
                                                              const LabelVectorSet* labelVectorSet) const = 0;

};

/**
 * Defines an interface for all classes that allow to configure a predictor that predicts label-wise probabilities for
 * given query examples.
 */
class IProbabilityPredictorConfig {

    public:

        virtual ~IProbabilityPredictorConfig() { };

        /**
         * Creates and returns a new object of type `IProbabilityPredictorFactory` according to the specified
         * configuration.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param numLabels     The total number of available labels
         * @return              An unique pointer to an object of type `IProbabilityPredictorFactory` that has been
         *                      created or a null pointer if the prediction of probabilities is not supported
         */
        virtual std::unique_ptr<IProbabilityPredictorFactory> createProbabilityPredictorFactory(
            const IFeatureMatrix& featureMatrix, uint32 numLabels) const = 0;

};
