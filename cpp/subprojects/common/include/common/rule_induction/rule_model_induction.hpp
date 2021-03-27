/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/input/nominal_feature_mask.hpp"
#include "common/input/feature_matrix.hpp"
#include "common/input/label_matrix.hpp"
#include "common/model/model_builder.hpp"
#include "common/sampling/random.hpp"


/**
 * Defines an interface for all classes that implement an algorithm for inducing several rules that will be added to a
 * resulting `RuleModel`.
 */
class IRuleModelInduction {

    public:

        virtual ~IRuleModelInduction() { };

        /**
         * Trains and returns a `RuleModel` that consists of several rules.
         *
         * @param nominalFeatureMaskPtr A shared pointer to an object of type `INominalFeatureMask` that provides access
         *                              to the information whether individual features are nominal or not
         * @param featureMatrixPtr      A shared pointer to an object of type `IFeatureMatrix` that provides access to
         *                              the feature values of the training examples
         * @param labelMatrixPtr        A shared pointer to an object of type `ILabelMatrix` that provides access to the
         *                              labels of the training examples
         * @param rng                   A reference to an object of type `RNG` that implements the random number
         *                              generator to be used
         * @param modelBuilder          A reference to an object of type `IModelBuilder`, the induced rules should be
         *                              added to
         * @return                      An unique pointer to an object of type `RuleModel` that consists of the rules
         *                              that have been induced
         */
        virtual std::unique_ptr<RuleModel> induceRules(std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr,
                                                       std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
                                                       std::shared_ptr<ILabelMatrix> labelMatrixPtr, RNG& rng,
                                                       IModelBuilder& modelBuilder) = 0;

};
