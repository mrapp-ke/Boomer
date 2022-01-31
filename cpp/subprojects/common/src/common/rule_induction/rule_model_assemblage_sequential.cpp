#include "common/rule_induction/rule_model_assemblage_sequential.hpp"


static inline IStoppingCriterion::Result testStoppingCriteria(
        std::forward_list<std::unique_ptr<IStoppingCriterion>>& stoppingCriteria, const IStatistics& statistics,
        uint32 numRules) {
    IStoppingCriterion::Result result;
    result.action = IStoppingCriterion::Action::CONTINUE;

    for (auto it = stoppingCriteria.begin(); it != stoppingCriteria.end(); it++) {
        std::unique_ptr<IStoppingCriterion>& stoppingCriterionPtr = *it;
        IStoppingCriterion::Result stoppingCriterionResult = stoppingCriterionPtr->test(statistics, numRules);
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
 * Allows to sequentially induce several rules, optionally starting with a default rule, that are added to a rule-based
 * model.
 */
class SequentialRuleModelAssemblage final : public IRuleModelAssemblage {

    private:

        std::unique_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr_;

        std::unique_ptr<IThresholdsFactory> thresholdsFactoryPtr_;

        std::unique_ptr<IRuleInductionFactory> ruleInductionFactoryPtr_;

        std::unique_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr_;

        std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr_;

        std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr_;

        std::unique_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr_;

        std::unique_ptr<IPruningFactory> pruningFactoryPtr_;

        std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr_;

        bool useDefaultRule_;

        std::forward_list<std::unique_ptr<IStoppingCriterionFactory>> stoppingCriterionFactories_;

    public:

        /**
         * @param statisticsProviderFactoryPtr  An unique pointer to an object of type `IStatisticsProviderFactory` that
         *                                      provides access to the statistics which serve as the basis for learning
         *                                      rules
         * @param thresholdsFactoryPtr          An unique pointer to an object of type `IThresholdsFactory` that allows
         *                                      to create objects that provide access to the thresholds that may be used
         *                                      by the conditions of rules
         * @param ruleInductionFactoryPtr       An unique pointer to an object of type `IRuleInductionFactory` that
         *                                      allows to create the implementation to be used for the induction of
         *                                      individual rules
         * @param labelSamplingFactoryPtr       An unique pointer to an object of type `ILabelSamplingFactory` that
         *                                      allows to create the implementation to be used for sampling the labels
         *                                      whenever a new rule is induced
         * @param instanceSamplingFactoryPtr    An unique pointer to an object of type `IInstanceSamplingFactory` that
         *                                      allows create the implementation to be used for sampling the examples
         *                                      whenever a new rule is induced
         * @param featureSamplingFactoryPtr     An unique pointer to an object of type `IFeatureSamplingFactory` that
         *                                      allows to create the implementation to be used for sampling the features
         *                                      that may be used by the conditions of a rule
         * @param partitionSamplingFactoryPtr   An unique pointer to an object of type `IPartitionSamplingFactory` that
         *                                      allows to create the implementation to be used for partitioning the
         *                                      training examples into a training set and a holdout set
         * @param pruningFactoryPtr             An unique pointer to an object of type `IPruningFactory` that allows to
         *                                      create the implementation to be used for pruning rules
         * @param postProcessorFactoryPtr       An unique pointer to an object of type `IPostProcessorFactory` that
         *                                      allows to create the implementation to be used for post-processing the
         *                                      predictions of rules
         * @param useDefaultRule                True, if a default rule should be used, False otherwise
         */
        SequentialRuleModelAssemblage(
            std::unique_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
            std::unique_ptr<IThresholdsFactory> thresholdsFactoryPtr,
            std::unique_ptr<IRuleInductionFactory> ruleInductionFactoryPtr,
            std::unique_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr,
            std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
            std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
            std::unique_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr,
            std::unique_ptr<IPruningFactory> pruningFactoryPtr,
            std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr,
            bool useDefaultRule)
            : statisticsProviderFactoryPtr_(std::move(statisticsProviderFactoryPtr)),
              thresholdsFactoryPtr_(std::move(thresholdsFactoryPtr)),
              ruleInductionFactoryPtr_(std::move(ruleInductionFactoryPtr)),
              labelSamplingFactoryPtr_(std::move(labelSamplingFactoryPtr)),
              instanceSamplingFactoryPtr_(std::move(instanceSamplingFactoryPtr)),
              featureSamplingFactoryPtr_(std::move(featureSamplingFactoryPtr)),
              partitionSamplingFactoryPtr_(std::move(partitionSamplingFactoryPtr)),
              pruningFactoryPtr_(std::move(pruningFactoryPtr)),
              postProcessorFactoryPtr_(std::move(postProcessorFactoryPtr)),
              useDefaultRule_(useDefaultRule) {

        }

        /**
         * Adds a new `IStoppingCriterionFactory` that allows to create the implementation of a stopping criterion that
         * should be used to decide whether the induction of additional rules should be stopped or not.
         *
         * @param stoppingCriterionFactoryPtr An unique pointer to an object of type `IStoppingCriterionFactory` that
         *                                    should be added
         */
        void addStoppingCriterionFactory(std::unique_ptr<IStoppingCriterionFactory> stoppingCriterionFactoryPtr) {
            stoppingCriterionFactories_.push_front(std::move(stoppingCriterionFactoryPtr));
        }

        std::unique_ptr<IRuleModel> induceRules(const INominalFeatureMask& nominalFeatureMask,
                                                const IColumnWiseFeatureMatrix& featureMatrix,
                                                const IRowWiseLabelMatrix& labelMatrix, uint32 randomState,
                                                IModelBuilder& modelBuilder) const override {
            uint32 numRules = useDefaultRule_ ? 1 : 0;
            uint32 numUsedRules = 0;

            // Partition training data...
            std::unique_ptr<IPartitionSampling> partitionSamplingPtr = labelMatrix.createPartitionSampling(
                *partitionSamplingFactoryPtr_);
            RNG rng(randomState);
            IPartition& partition = partitionSamplingPtr->partition(rng);

            // Initialize stopping criteria...
            std::forward_list<std::unique_ptr<IStoppingCriterion>> stoppingCriteria;

            for (auto it = stoppingCriterionFactories_.cbegin(); it != stoppingCriterionFactories_.cend(); it++) {
                const std::unique_ptr<IStoppingCriterionFactory>& stoppingCriterionFactoryPtr = *it;
                stoppingCriteria.push_front(partition.createStoppingCriterion(*stoppingCriterionFactoryPtr));
            }

            // Induce default rule...
            std::unique_ptr<IStatisticsProvider> statisticsProviderPtr = labelMatrix.createStatisticsProvider(
                *statisticsProviderFactoryPtr_);
            std::unique_ptr<IRuleInduction> ruleInductionPtr = ruleInductionFactoryPtr_->create();

            if (useDefaultRule_) {
                ruleInductionPtr->induceDefaultRule(statisticsProviderPtr->get(), modelBuilder);
            }

            statisticsProviderPtr->switchToRegularRuleEvaluation();

            // Induce the remaining rules...
            std::unique_ptr<IThresholds> thresholdsPtr = thresholdsFactoryPtr_->create(featureMatrix,
                                                                                       nominalFeatureMask,
                                                                                       *statisticsProviderPtr);
            std::unique_ptr<IInstanceSampling> instanceSamplingPtr = partition.createInstanceSampling(
                *instanceSamplingFactoryPtr_, labelMatrix, statisticsProviderPtr->get());
            std::unique_ptr<IFeatureSampling> featureSamplingPtr = featureSamplingFactoryPtr_->create();
            std::unique_ptr<ILabelSampling> labelSamplingPtr = labelSamplingFactoryPtr_->create();
            std::unique_ptr<IPruning> pruningPtr = pruningFactoryPtr_->create();
            std::unique_ptr<IPostProcessor> postProcessorPtr = postProcessorFactoryPtr_->create();
            IStoppingCriterion::Result stoppingCriterionResult;

            while (stoppingCriterionResult = testStoppingCriteria(stoppingCriteria, statisticsProviderPtr->get(),
                                                                  numRules),
                   stoppingCriterionResult.action != IStoppingCriterion::Action::FORCE_STOP) {
                if (stoppingCriterionResult.action == IStoppingCriterion::Action::STORE_STOP && numUsedRules == 0) {
                    numUsedRules = stoppingCriterionResult.numRules;
                }

                const IWeightVector& weights = instanceSamplingPtr->sample(rng);
                const IIndexVector& labelIndices = labelSamplingPtr->sample(rng);
                bool success = ruleInductionPtr->induceRule(*thresholdsPtr, labelIndices, weights, partition,
                                                            *featureSamplingPtr, *pruningPtr, *postProcessorPtr, rng,
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

/**
 * A factory that allows to create instances of the class `IRuleModelAssemblage` that allow to sequentially induce
 * several rules, optionally starting with a default rule, that are added to a rule-based model.
 */
class SequentialRuleModelAssemblageFactory final : public IRuleModelAssemblageFactory {

    private:

        bool useDefaultRule_;

    public:

        /**
         * @param useDefaultRule True, if a default rule should be used, false otherwise
         */
        SequentialRuleModelAssemblageFactory(bool useDefaultRule)
            : useDefaultRule_(useDefaultRule) {

        }

        std::unique_ptr<IRuleModelAssemblage> create(
                std::unique_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
                std::unique_ptr<IThresholdsFactory> thresholdsFactoryPtr,
                std::unique_ptr<IRuleInductionFactory> ruleInductionFactoryPtr,
                std::unique_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr,
                std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
                std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
                std::unique_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr,
                std::unique_ptr<IPruningFactory> pruningFactoryPtr,
                std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr,
                std::forward_list<std::unique_ptr<IStoppingCriterionFactory>>& stoppingCriterionFactories) const override {
            std::unique_ptr<SequentialRuleModelAssemblage> rule_model_assemblage_ptr =
                std::make_unique<SequentialRuleModelAssemblage>(
                    std::move(statisticsProviderFactoryPtr), std::move(thresholdsFactoryPtr),
                    std::move(ruleInductionFactoryPtr), std::move(labelSamplingFactoryPtr),
                    std::move(instanceSamplingFactoryPtr), std::move(featureSamplingFactoryPtr),
                    std::move(partitionSamplingFactoryPtr), std::move(pruningFactoryPtr),
                    std::move(postProcessorFactoryPtr), useDefaultRule_);

            for (auto it = stoppingCriterionFactories.begin(); it != stoppingCriterionFactories.end(); it++) {
                rule_model_assemblage_ptr->addStoppingCriterionFactory(std::move(*it));
            }

            return rule_model_assemblage_ptr;
        }

};

SequentialRuleModelAssemblageConfig::SequentialRuleModelAssemblageConfig()
    : useDefaultRule_(true) {

}

bool SequentialRuleModelAssemblageConfig::getUseDefaultRule() const {
    return useDefaultRule_;
}

ISequentialRuleModelAssemblageConfig& SequentialRuleModelAssemblageConfig::setUseDefaultRule(bool useDefaultRule) {
    useDefaultRule_ = useDefaultRule;
    return *this;
}

std::unique_ptr<IRuleModelAssemblageFactory> SequentialRuleModelAssemblageConfig::createRuleModelAssemblageFactory() const {
    return std::make_unique<SequentialRuleModelAssemblageFactory>(useDefaultRule_);
}
