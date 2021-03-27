#include "common/rule_induction/rule_model_induction_sequential.hpp"


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

SequentialRuleModelInduction::SequentialRuleModelInduction(
        std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
        std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr, std::shared_ptr<IRuleInduction> ruleInductionPtr,
        std::shared_ptr<IHeadRefinementFactory> defaultRuleHeadRefinementFactoryPtr,
        std::shared_ptr<IHeadRefinementFactory> headRefinementFactoryPtr,
        std::shared_ptr<ILabelSubSampling> labelSubSamplingPtr,
        std::shared_ptr<IInstanceSubSampling> instanceSubSamplingPtr,
        std::shared_ptr<IFeatureSubSampling> featureSubSamplingPtr,
        std::shared_ptr<IPartitionSampling> partitionSamplingPtr, std::shared_ptr<IPruning> pruningPtr,
        std::shared_ptr<IPostProcessor> postProcessorPtr, uint32 minCoverage, intp maxConditions,
        intp maxHeadRefinements,
        std::unique_ptr<std::forward_list<std::shared_ptr<IStoppingCriterion>>> stoppingCriteriaPtr)
    : statisticsProviderFactoryPtr_(statisticsProviderFactoryPtr), thresholdsFactoryPtr_(thresholdsFactoryPtr),
      ruleInductionPtr_(ruleInductionPtr), defaultRuleHeadRefinementFactoryPtr_(defaultRuleHeadRefinementFactoryPtr),
      headRefinementFactoryPtr_(headRefinementFactoryPtr), labelSubSamplingPtr_(labelSubSamplingPtr),
      instanceSubSamplingPtr_(instanceSubSamplingPtr), featureSubSamplingPtr_(featureSubSamplingPtr),
      partitionSamplingPtr_(partitionSamplingPtr), pruningPtr_(pruningPtr), postProcessorPtr_(postProcessorPtr),
      minCoverage_(minCoverage), maxConditions_(maxConditions), maxHeadRefinements_(maxHeadRefinements),
      stoppingCriteriaPtr_(std::move(stoppingCriteriaPtr)) {

}

std::unique_ptr<RuleModel> SequentialRuleModelInduction::induceRules(
        std::shared_ptr<INominalFeatureMask> nominalFeatureMaskPtr, std::shared_ptr<IFeatureMatrix> featureMatrixPtr,
        std::shared_ptr<ILabelMatrix> labelMatrixPtr, RNG& rng, IModelBuilder& modelBuilder) {
    // Induce default rule...
    const IHeadRefinementFactory* defaultRuleHeadRefinementFactory = defaultRuleHeadRefinementFactoryPtr_.get();
    uint32 numRules = defaultRuleHeadRefinementFactory != nullptr ? 1 : 0;
    uint32 numUsedRules = 0;
     std::shared_ptr<IRandomAccessLabelMatrix> randomAccessLabelMatrixPtr =
        std::dynamic_pointer_cast<IRandomAccessLabelMatrix, ILabelMatrix>(labelMatrixPtr);
    std::shared_ptr<IStatisticsProvider> statisticsProviderPtr = statisticsProviderFactoryPtr_->create(
        randomAccessLabelMatrixPtr);
    ruleInductionPtr_->induceDefaultRule(*statisticsProviderPtr, defaultRuleHeadRefinementFactory, modelBuilder);

    // Induce the remaining rules...
    std::unique_ptr<IThresholds> thresholdsPtr = thresholdsFactoryPtr_->create(featureMatrixPtr, nominalFeatureMaskPtr,
                                                                               statisticsProviderPtr,
                                                                               headRefinementFactoryPtr_);
    uint32 numExamples = thresholdsPtr->getNumExamples();
    uint32 numLabels = thresholdsPtr->getNumLabels();
    std::unique_ptr<IPartition> partitionPtr = partitionSamplingPtr_->partition(numExamples, rng);
    IStoppingCriterion::Result stoppingCriterionResult;

    while (stoppingCriterionResult = testStoppingCriteria(*stoppingCriteriaPtr_, *partitionPtr,
                                                          statisticsProviderPtr->get(),
                                                          numRules),
           stoppingCriterionResult.action != IStoppingCriterion::Action::FORCE_STOP) {
        if (stoppingCriterionResult.action == IStoppingCriterion::Action::STORE_STOP && numUsedRules == 0) {
            numUsedRules = stoppingCriterionResult.numRules;
        }

        std::unique_ptr<IWeightVector> weightsPtr = partitionPtr->subSample(*instanceSubSamplingPtr_, rng);
        std::unique_ptr<IIndexVector> labelIndicesPtr = labelSubSamplingPtr_->subSample(numLabels, rng);
        bool success = ruleInductionPtr_->induceRule(*thresholdsPtr, *labelIndicesPtr, *weightsPtr, *partitionPtr,
                                                     *featureSubSamplingPtr_, *pruningPtr_, *postProcessorPtr_,
                                                     minCoverage_, maxConditions_, maxHeadRefinements_, rng,
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
