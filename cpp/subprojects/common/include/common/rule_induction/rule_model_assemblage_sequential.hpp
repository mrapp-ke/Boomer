/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/rule_induction/rule_model_assemblage.hpp"


/**
 * A factory that allows to create instances of the class `SequentialRuleModelAssemblage`.
 */
class SequentialRuleModelAssemblageFactory final : public IRuleModelAssemblageFactory {

    public:

        std::unique_ptr<IRuleModelAssemblage> create(
            std::shared_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
            std::shared_ptr<IThresholdsFactory> thresholdsFactoryPtr, std::shared_ptr<IRuleInduction> ruleInductionPtr,
            std::shared_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr,
            std::shared_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
            std::shared_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
            std::shared_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr,
            std::shared_ptr<IPruning> pruningPtr, std::shared_ptr<IPostProcessor> postProcessorPtr,
            const std::forward_list<std::shared_ptr<IStoppingCriterion>> stoppingCriteria,
            bool useDefaultRule) const override;

};
