/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/nominal_feature_mask.hpp"
#include "common/input/feature_matrix.hpp"
#include "common/input/label_matrix.hpp"
#include "common/model/model_builder.hpp"
#include "common/rule_induction/rule_induction.hpp"
#include "common/sampling/label_sampling.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/sampling/feature_sampling.hpp"
#include "common/sampling/partition_sampling.hpp"
#include "common/statistics/statistics_provider_factory.hpp"
#include "common/stopping/stopping_criterion.hpp"
#include "common/thresholds/thresholds_factory.hpp"
#include <forward_list>


/**
 * Defines an interface for all classes that implement an algorithm for inducing several rules that will be added to a
 * resulting `RuleModel`.
 */
class IRuleModelAssemblage {

    public:

        virtual ~IRuleModelAssemblage() { };

        /**
         * Assembles and returns a `RuleModel` that consists of several rules.
         *
         * @param nominalFeatureMask    A reference to an object of type `INominalFeatureMask` that provides access to
         *                              the information whether individual features are nominal or not
         * @param featureMatrix         A reference to an object of type `IFeatureMatrix` that provides access to the
         *                              feature values of the training examples
         * @param labelMatrix           A reference to an object of type `ILabelMatrix` that provides access to the
         *                              labels of the training examples
         * @param randomState           The seed to be used by the random number generators
         * @param modelBuilder          A reference to an object of type `IModelBuilder`, the induced rules should be
         *                              added to
         * @return                      An unique pointer to an object of type `RuleModel` that consists of the rules
         *                              that have been induced
         */
        virtual std::unique_ptr<RuleModel> induceRules(const INominalFeatureMask& nominalFeatureMask,
                                                       const IFeatureMatrix& featureMatrix,
                                                       const ILabelMatrix& labelMatrix, uint32 randomState,
                                                       IModelBuilder& modelBuilder) = 0;

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IRuleModelAssemblage`.
 */
class IRuleModelAssemblageFactory {

    public:

        virtual ~IRuleModelAssemblageFactory() { };

        /**
         * Creates and returns a new object of the type `IRuleModelAssemblage`.
         *
         * @param statisticsProviderFactoryPtr  A shared pointer to an object of type `IStatisticsProviderFactory` that
         *                                      provides access to the statistics which serve as the basis for learning
         *                                      rules
         * @param thresholdsFactoryPtr          A shared pointer to an object of type `IThresholdsFactory` that allows
         *                                      to create objects that provide access to the thresholds that may be used
         *                                      by the conditions of rules
         * @param ruleInductionPtr              A shared pointer to an object of type `IRuleInduction` that should be
         *                                      used to induce individual rules
         * @param labelSamplingFactoryPtr       A shared pointer to an object of type `ILabelSamplingFactory` that
         *                                      allows to create the implementation to be used for sampling the labels
         *                                      whenever a new rule is induced
         * @param instanceSamplingFactoryPtr    A shared pointer to an object of type `IInstanceSamplingFactory` that
         *                                      allows create the implementation to be used for sampling the examples
         *                                      whenever a new rule is induced
         * @param featureSamplingFactoryPtr     A shared pointer to an object of type `IFeatureSamplingFactory` that
         *                                      allows to create the implementation to be used for sampling the features
         *                                      that may be used by the conditions of a rule
         * @param partitionSamplingFactoryPtr   A shared pointer to an object of type `IPartitionSamplingFactory` that
         *                                      allows to create the implementation to be used for partitioning the
         *                                      training examples into a training set and a holdout set
         * @param pruningPtr                    A shared pointer to an object of type `IPruning` that should be used to
         *                                      prune the rules
         * @param postProcessorPtr              A shared pointer to an object of type `IPostProcessor` that should be
         *                                      used to post-process the predictions of rules
         * @param stoppingCriteria              A list that stores the stopping criteria, which should be used to decide
         *                                      whether additional rules should be induced or not
         * @param useDefaultRule                True, if a default rule should be used, false otherwise
         */
        virtual std::unique_ptr<IRuleModelAssemblage> create(
            std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
            std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr, std::shared_ptr<IRuleInduction> ruleInductionPtr,
            std::shared_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr,
            std::shared_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
            std::shared_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
            std::shared_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr,
            std::shared_ptr<IPruning> pruningPtr, std::shared_ptr<IPostProcessor> postProcessorPtr,
            const std::forward_list<std::shared_ptr<IStoppingCriterion>> stoppingCriteria,
            bool useDefaultRule) const = 0;

};
