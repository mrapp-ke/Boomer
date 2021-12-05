/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_induction/rule_model_assemblage.hpp"


/**
 * A builder that allows to configure rule learning algorithms.
 */
class AlgorithmBuilder final {

    private:

        std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr_;

        std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr_;

        std::shared_ptr<IRuleInduction> ruleInductionPtr_;

        std::shared_ptr<IRuleModelAssemblageFactory> ruleModelAssemblageFactoryPtr_;

        std::shared_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr_;

        std::shared_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr_;

        std::shared_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr_;

        std::shared_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr_;

        std::shared_ptr<IPruning> pruningPtr_;

        std::shared_ptr<IPostProcessor> postProcessorPtr_;

        std::forward_list<std::shared_ptr<IStoppingCriterion>> stoppingCriteria_;

        bool useDefaultRule_;

    public:

        /**
         * @param statisticsProviderFactoryPtr  An unique pointer to an object of type `IStatisticsProviderFactory` to
         *                                      be used by the rule learner to access the statistics that serve as the
         *                                      basis for learning rules
         * @param thresholdsFactoryPtr          An unique pointer to an object of type `IThresholdsFactory` to be used
         *                                      by the rule learner to access the thresholds that may be used by the
         *                                      conditions of rules
         * @param ruleInductionPtr              An unique pointer to an object of type `IRuleInduction` to be used by
         *                                      the rule learner to induce individual rules
         * @param ruleModelAssemblageFactoryPtr An unique pointer to an object of type `IRuleModelAssemblageFactory` to
         *                                      be used by the rule learner for the assemblage of a rule model
         */
        AlgorithmBuilder(std::unique_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
                         std::unique_ptr<IThresholdsFactory> thresholdsFactoryPtr,
                         std::unique_ptr<IRuleInduction> ruleInductionPtr,
                         std::unique_ptr<IRuleModelAssemblageFactory> ruleModelAssemblageFactoryPtr);

        /**
         * Sets whether a default rule should be used or not.
         *
         * @param useDefaultRule    True, if a default rule should be used, false otherwise.
         * @return                  A reference to the builder itself
         */
        AlgorithmBuilder& setUseDefaultRule(bool useDefaultRule);

        /**
         * Sets the `ILabelSamplingFactory` to be used by the rule learner to sample the labels individual rules may
         * predict for.
         *
         * @param labelSamplingFactoryPtr   An unique pointer to an object of type `ILabelSamplingFactory` to be set
         * @return                          A reference to the builder itself
         */
        AlgorithmBuilder& setLabelSamplingFactory(std::unique_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr);

        /**
         * Sets the `IInstanceSamplingFactory` to be used by the rule learner to sample the instances whenever a new
         * rule is induced.
         *
         * @param instanceSamplingFactoryPtr    An unique pointer to an object of type `IInstanceSamplingFactory` to be
         *                                      set
         * @return                              A reference to the builder itself
         */
        AlgorithmBuilder& setInstanceSamplingFactory(
            std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr);

        /**
         * Sets the `IFeatureSamplingFactory` to be used by the rule learner to sample the features whenever a rule
         * should be refined.
         *
         * @param featureSamplingFactoryPtr An unique pointer to an object of type `IFeatureSamplingFactory` to be set
         * @return                          A reference to the builder itself
         */
        AlgorithmBuilder& setFeatureSamplingFactory(std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr);

        /**
         * Sets the `IPartitionSamplingFactory` to be used by the rule learner to create a holdout set.
         *
         * @param partitionSamplingFactoryPtr   An unique pointer to an object of type `IPartitionSamplingFactory` to be
         *                                      set
         * @return                              A reference to the builder itself
         */
        AlgorithmBuilder& setPartitionSamplingFactory(
            std::unique_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr);

        /**
         * Sets the `IPruning` to be used by the rule learner to prune individual rules.
         *
         * @param pruningPtr    An unique pointer to an object of type `IPruning` to be set
         * @return              A reference to the builder itself
         */
        AlgorithmBuilder& setPruning(std::unique_ptr<IPruning> pruningPtr);

        /**
         * Sets the `IPostProcessor` to be used by the rule learner to post-process the predictions of individual rules.
         *
         * @param postProcessorPtr  An unique pointer to an object of type `IPostProcessor` to be set
         * @return                  A reference to the builder itself
         */
        AlgorithmBuilder& setPostProcessor(std::unique_ptr<IPostProcessor> postProcessorPtr);

        /**
         * Adds a `IStoppingCriterion` that should be used by the rule learner to decide when the induction of
         * additional rules should be stopped.
         *
         * @param stoppingCriterionPtr  An unique pointer to an object of type `IStoppingCriterion` to be added
         * @return                      A reference to the builder itself
         */
        AlgorithmBuilder& addStoppingCriterion(std::unique_ptr<IStoppingCriterion> stoppingCriterionPtr);

        /**
         * Creates and returns a new object of type `IRuleModelAssemblage`.
         *
         * @return An unique pointer to an object of type `IRuleModelAssemblage` that has been created
         */
        std::unique_ptr<IRuleModelAssemblage> build() const;

};
