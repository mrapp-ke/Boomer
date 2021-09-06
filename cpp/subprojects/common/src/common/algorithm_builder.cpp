#include "common/algorithm_builder.hpp"
#include "common/post_processing/post_processor_no.hpp"
#include "common/pruning/pruning_no.hpp"
#include "common/sampling/feature_sampling_no.hpp"
#include "common/sampling/instance_sampling_no.hpp"
#include "common/sampling/label_sampling_no.hpp"
#include "common/sampling/partition_sampling_no.hpp"
#include "common/validation.hpp"


AlgorithmBuilder::AlgorithmBuilder(std::unique_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
                                   std::unique_ptr<IThresholdsFactory> thresholdsFactoryPtr,
                                   std::unique_ptr<IRuleInduction> ruleInductionPtr,
                                   std::unique_ptr<IRuleModelAssemblageFactory> ruleModelAssemblageFactoryPtr)
    : statisticsProviderFactoryPtr_(std::move(statisticsProviderFactoryPtr)),
      thresholdsFactoryPtr_(std::move(thresholdsFactoryPtr)), ruleInductionPtr_(std::move(ruleInductionPtr)),
      ruleModelAssemblageFactoryPtr_(std::move(ruleModelAssemblageFactoryPtr)),
      labelSamplingFactoryPtr_(std::make_shared<NoLabelSamplingFactory>()),
      instanceSamplingFactoryPtr_(std::make_shared<NoInstanceSamplingFactory>()),
      featureSamplingFactoryPtr_(std::make_shared<NoFeatureSamplingFactory>()),
      partitionSamplingFactoryPtr_(std::make_shared<NoPartitionSamplingFactory>()),
      pruningPtr_(std::make_shared<NoPruning>()), postProcessorPtr_(std::make_shared<NoPostProcessor>()),
      useDefaultRule_(true) {
    assertNotNull("statisticsProviderFactoryPtr", statisticsProviderFactoryPtr_.get());
    assertNotNull("thresholdsFactoryPtr", thresholdsFactoryPtr_.get());
    assertNotNull("ruleInductionPtr", ruleInductionPtr_.get());
    assertNotNull("ruleModelAssemblageFactoryPtr", ruleModelAssemblageFactoryPtr_.get());
}

AlgorithmBuilder& AlgorithmBuilder::setUseDefaultRule(bool useDefaultRule) {
    useDefaultRule_ = useDefaultRule;
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setLabelSamplingFactory(
        std::unique_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr) {
    assertNotNull("labelSamplingFactoryPtr", labelSamplingFactoryPtr.get());
    labelSamplingFactoryPtr_ = std::move(labelSamplingFactoryPtr);
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setInstanceSamplingFactory(
        std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr) {
    assertNotNull("instanceSamplingFactoryPtr", instanceSamplingFactoryPtr.get());
    instanceSamplingFactoryPtr_ = std::move(instanceSamplingFactoryPtr);
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setFeatureSamplingFactory(
        std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr) {
    assertNotNull("featureSamplingFactoryPtr", featureSamplingFactoryPtr.get());
    featureSamplingFactoryPtr_ = std::move(featureSamplingFactoryPtr);
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setPartitionSamplingFactory(
        std::unique_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr) {
    assertNotNull("partitionSamplingFactoryPtr", partitionSamplingFactoryPtr.get());
    partitionSamplingFactoryPtr_ = std::move(partitionSamplingFactoryPtr);
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setPruning(std::unique_ptr<IPruning> pruningPtr) {
    assertNotNull("pruningPtr", pruningPtr.get());
    pruningPtr_ = std::move(pruningPtr);
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::setPostProcessor(std::unique_ptr<IPostProcessor> postProcessorPtr) {
    assertNotNull("postProcessorPtr", postProcessorPtr.get());
    postProcessorPtr_ = std::move(postProcessorPtr);
    return *this;
}

AlgorithmBuilder& AlgorithmBuilder::addStoppingCriterion(std::unique_ptr<IStoppingCriterion> stoppingCriterionPtr) {
    assertNotNull("stoppingCriterionPtr", stoppingCriterionPtr.get());
    stoppingCriteria_.push_front(std::move(stoppingCriterionPtr));
    return *this;
}

std::unique_ptr<IRuleModelAssemblage> AlgorithmBuilder::build() const {
    return ruleModelAssemblageFactoryPtr_->create(statisticsProviderFactoryPtr_, thresholdsFactoryPtr_,
                                                  ruleInductionPtr_, labelSamplingFactoryPtr_,
                                                  instanceSamplingFactoryPtr_, featureSamplingFactoryPtr_,
                                                  partitionSamplingFactoryPtr_, pruningPtr_, postProcessorPtr_,
                                                  stoppingCriteria_, useDefaultRule_);
}
