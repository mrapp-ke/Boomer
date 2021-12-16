/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/model/model_builder.hpp"
#include "common/post_processing/post_processor.hpp"
#include "common/pruning/pruning.hpp"
#include "common/sampling/feature_sampling.hpp"
#include "common/sampling/weight_vector.hpp"
#include "common/sampling/partition.hpp"
#include "common/statistics/statistics.hpp"
#include "common/thresholds/thresholds.hpp"



/**
 * Defines an interface for all classes that implement an algorithm for inducing individual rules.
 */
class IRuleInduction {

    public:

        virtual ~IRuleInduction() { };

        /**
         * Induces the default rule.
         *
         * @param statistics    A reference to an object of type `IStatistics` that provides access to the statistics
         *                      which should serve as the basis for inducing the default rule
         * @param modelBuilder  A reference to an object of type `IModelBuilder`, the default rule should be added to
         */
        virtual void induceDefaultRule(IStatistics& statistics, IModelBuilder& modelBuilder) const = 0;

        /**
         * Induces a new rule.
         *
         * @param thresholds        A reference to an object of type `IThresholds` that provides access to the
         *                          thresholds that may be used by the conditions of the rule
         * @param labelIndices      A reference to an object of type `IIndexVector` that provides access to the indices
         *                          of the labels for which the rule may predict
         * @param weights           A reference to an object of type `IWeightVector` that provides access to the weights
         *                          of individual training examples
         * @param partition         A reference to an object of type `IPartition` that provides access to the indices of
         *                          the training examples that belong to the training set and the holdout set,
         *                          respectively
         * @param featureSampling   A reference to an object of type `IFeatureSampling` that should be used for sampling
         *                          the features that may be used by a new condition
         * @param pruning           A reference to an object of type `IPruning` that should be used to prune the rule
         * @param postProcessor     A reference to an object of type `IPostProcessor` that should be used to
         *                          post-process the predictions of the rule
         * @param rng               A reference to an object of type `RNG` that implements the random number generator
         *                          to be used
         * @param modelBuilder      A reference to an object of type `IModelBuilder`, the rule should be added to
         * @return                  True, if a rule has been induced, false otherwise
         */
        virtual bool induceRule(IThresholds& thresholds, const IIndexVector& labelIndices,
                                const IWeightVector& weights, IPartition& partition, IFeatureSampling& featureSampling,
                                const IPruning& pruning, const IPostProcessor& postProcessor, RNG& rng,
                                IModelBuilder& modelBuilder) const = 0;

};
