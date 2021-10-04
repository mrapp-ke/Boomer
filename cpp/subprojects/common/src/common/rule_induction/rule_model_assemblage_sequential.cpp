#include "common/rule_induction/rule_model_assemblage_sequential.hpp"


static inline IStoppingCriterion::Result testStoppingCriteria(
        std::forward_list<std::shared_ptr<IStoppingCriterion>>& stoppingCriteria, const IPartition& partition,
        const IStatistics& statistics, uint32 numRules) {
    IStoppingCriterion::Result result;
    result.action = IStoppingCriterion::Action::CONTINUE;

    for (auto it = stoppingCriteria.begin(); it != stoppingCriteria.end(); it++) {
        std::shared_ptr<IStoppingCriterion>& stoppingCriterionPtr = *it;
        IStoppingCriterion::Result stoppingCriterionResult = stoppingCriterionPtr->test(partition, statistics,
                                                                                        numRules);
        IStoppingCriterion::Action action = stoppingCriterionResult.action;

        switch (action) {
            case IStoppingCriterion::Action::FORCE_STOP: {
                result.action = action;
                result.numRules = stoppingCriterionResult.numRules;
                return result;
            }
            case IStoppingCriterion::Action::STORE_STOP: {
                result.action = action;
                result.numRules = stoppingCriterionResult.numRules;
                break;
            }
            default: {
                break;
            }
        }
    }

    return result;
}

/**
 * Allows to sequentially induce several rules, starting with a default rule, that will be added to a resulting
 * `RuleModel`.
 */
class SequentialRuleModelAssemblage : public IRuleModelAssemblage {

    private:

        std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr_;

        std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr_;

        std::shared_ptr<IRuleInduction> ruleInductionPtr_;

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
         * @param statisticsProviderFactoryPtr          A shared pointer to an object of type
         *                                              `IStatisticsProviderFactory` that provides access to the
         *                                              statistics which serve as the basis for learning rules
         * @param thresholdsFactoryPtr                  A shared pointer to an object of type `IThresholdsFactory` that
         *                                              allows to create objects that provide access to the thresholds
         *                                              that may be used by the conditions of rules
         * @param ruleInductionPtr                      A shared pointer to an object of type `IRuleInduction` that
         *                                              should be used to induce individual rules
         * @param labelSamplingFactoryPtr               A shared pointer to an object of type `ILabelSamplingFactory`
         *                                              that allows to create the implementation to be used for sampling
         *                                              the labels whenever a new rule is induced
         * @param instanceSamplingFactoryPtr            A shared pointer to an object of type `IInstanceSamplingFactory`
         *                                              that allows create the implementation to be used for sampling
         *                                              the examples whenever a new rule is induced
         * @param featureSamplingFactoryPtr             A shared pointer to an object of type `IFeatureSamplingFactory`
         *                                              that allows to create the implementation to be used for sampling
         *                                              the features that may be used by the conditions of a rule
         * @param partitionSamplingFactoryPtr           A shared pointer to an object of type
         *                                              `IPartitionSamplingFactory` that allows to create the
         *                                              implementation to be used for partitioning the training examples
         *                                              into a training set and a holdout set
         * @param pruningPtr                            A shared pointer to an object of type `IPruning` that should be
         *                                              used to prune the rules
         * @param postProcessorPtr                      A shared pointer to an object of type `IPostProcessor` that
         *                                              should be used to post-process the predictions of rules
         * @param stoppingCriteriaPtr                   A list that contains the stopping criteria, which should be used
         *                                              to decide whether additional rules should be induced or not
         * @param useDefaultRule                        True, if a default rule should be used, False otherwise
         */
        SequentialRuleModelAssemblage(
            std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
            std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr, std::shared_ptr<IRuleInduction> ruleInductionPtr,
            std::shared_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr,
            std::shared_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
            std::shared_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
            std::shared_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr,
            std::shared_ptr<IPruning> pruningPtr, std::shared_ptr<IPostProcessor> postProcessorPtr,
            std::forward_list<std::shared_ptr<IStoppingCriterion>> stoppingCriteria, bool useDefaultRule)
        : statisticsProviderFactoryPtr_(statisticsProviderFactoryPtr), thresholdsFactoryPtr_(thresholdsFactoryPtr),
          ruleInductionPtr_(ruleInductionPtr), labelSamplingFactoryPtr_(labelSamplingFactoryPtr),
          instanceSamplingFactoryPtr_(instanceSamplingFactoryPtr),
          featureSamplingFactoryPtr_(featureSamplingFactoryPtr),
          partitionSamplingFactoryPtr_(partitionSamplingFactoryPtr), pruningPtr_(pruningPtr),
          postProcessorPtr_(postProcessorPtr), stoppingCriteria_(stoppingCriteria), useDefaultRule_(useDefaultRule) {

        }

        std::unique_ptr<RuleModel> induceRules(const INominalFeatureMask& nominalFeatureMask,
                                               const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix,
                                               uint32 randomState, IModelBuilder& modelBuilder) {
            // Induce default rule...
            uint32 numRules = useDefaultRule_ ? 1 : 0;
            uint32 numUsedRules = 0;
            std::unique_ptr<IStatisticsProvider> statisticsProviderPtr = labelMatrix.createStatisticsProvider(
                *statisticsProviderFactoryPtr_);

            if (useDefaultRule_) {
                ruleInductionPtr_->induceDefaultRule(statisticsProviderPtr->get(), modelBuilder);
            }

            statisticsProviderPtr->switchToRegularRuleEvaluation();

            // Induce the remaining rules...
            std::unique_ptr<IThresholds> thresholdsPtr = thresholdsFactoryPtr_->create(featureMatrix,
                                                                                       nominalFeatureMask,
                                                                                       *statisticsProviderPtr);
            uint32 numFeatures = thresholdsPtr->getNumFeatures();
            uint32 numLabels = thresholdsPtr->getNumLabels();
            std::unique_ptr<IPartitionSampling> partitionSamplingPtr = labelMatrix.createPartitionSampling(
                *partitionSamplingFactoryPtr_);
            RNG rng(randomState);
            IPartition& partition = partitionSamplingPtr->partition(rng);
            std::unique_ptr<IInstanceSampling> instanceSamplingPtr = partition.createInstanceSampling(
                *instanceSamplingFactoryPtr_, labelMatrix, statisticsProviderPtr->get());
            std::unique_ptr<IFeatureSampling> featureSamplingPtr = featureSamplingFactoryPtr_->create(numFeatures);
            std::unique_ptr<ILabelSampling> labelSamplingPtr = labelSamplingFactoryPtr_->create(numLabels);
            IStoppingCriterion::Result stoppingCriterionResult;

            while (stoppingCriterionResult = testStoppingCriteria(stoppingCriteria_, partition,
                                                                  statisticsProviderPtr->get(), numRules),
                   stoppingCriterionResult.action != IStoppingCriterion::Action::FORCE_STOP) {
                if (stoppingCriterionResult.action == IStoppingCriterion::Action::STORE_STOP && numUsedRules == 0) {
                    numUsedRules = stoppingCriterionResult.numRules;
                }

                const IWeightVector& weights = instanceSamplingPtr->sample(rng);
                const IIndexVector& labelIndices = labelSamplingPtr->sample(rng);
                bool success = ruleInductionPtr_->induceRule(*thresholdsPtr, labelIndices, weights, partition,
                                                             *featureSamplingPtr, *pruningPtr_, *postProcessorPtr_, rng,
                                                             modelBuilder);

                if (success) {
                    numRules++;
                } else {
                    break;
                }
            }

            // Build and return the final model...
            return modelBuilder.build(numUsedRules);
        }

};

std::unique_ptr<IRuleModelAssemblage> SequentialRuleModelAssemblageFactory::create(
        std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
        std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr, std::shared_ptr<IRuleInduction> ruleInductionPtr,
        std::shared_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr,
        std::shared_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
        std::shared_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
        std::shared_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr,
        std::shared_ptr<IPruning> pruningPtr, std::shared_ptr<IPostProcessor> postProcessorPtr,
        const std::forward_list<std::shared_ptr<IStoppingCriterion>> stoppingCriteria, bool useDefaultRule) const {
    return std::make_unique<SequentialRuleModelAssemblage>(statisticsProviderFactoryPtr, thresholdsFactoryPtr,
                                                           ruleInductionPtr, labelSamplingFactoryPtr,
                                                           instanceSamplingFactoryPtr, featureSamplingFactoryPtr,
                                                           partitionSamplingFactoryPtr, pruningPtr, postProcessorPtr,
                                                           stoppingCriteria, useDefaultRule);
}
