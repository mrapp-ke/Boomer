#include "common/post_optimization/post_optimization_phase_list.hpp"

/**
 * An implementation of the class `IPostOptimization` that carries out several post-optimization phases.
 */
class PostOptimizationPhaseList final : public IPostOptimization {
    private:

        const std::unique_ptr<IntermediateModelBuilder> intermediateModelBuilderPtr_;

        std::vector<std::unique_ptr<IPostOptimizationPhase>> postOptimizationPhases_;

    public:

        /**
         * @param modelBuilderPtr                   An unique pointer to an object of type `IModelBuilder` that should
         *                                          be used to build the final model
         * @param postOptimizationPhaseFactories    A reference to a vector that stores the factories that allow to
         *                                          create instances of the optimization phases to be carried out
         */
        PostOptimizationPhaseList(
          std::unique_ptr<IModelBuilder> modelBuilderPtr,
          const std::vector<std::unique_ptr<IPostOptimizationPhaseFactory>>& postOptimizationPhaseFactories)
            : intermediateModelBuilderPtr_(std::make_unique<IntermediateModelBuilder>(std::move(modelBuilderPtr))) {
            postOptimizationPhases_.reserve(postOptimizationPhaseFactories.size());

            for (auto it = postOptimizationPhaseFactories.cbegin(); it != postOptimizationPhaseFactories.cend(); it++) {
                const std::unique_ptr<IPostOptimizationPhaseFactory>& postOptimizationPhaseFactoryPtr = *it;
                std::unique_ptr<IPostOptimizationPhase> postOptimizationPhasePtr =
                  postOptimizationPhaseFactoryPtr->create(*intermediateModelBuilderPtr_);
                postOptimizationPhases_.push_back(std::move(postOptimizationPhasePtr));
            }
        }

        IModelBuilder& getModelBuilder() const override {
            return *intermediateModelBuilderPtr_;
        }

        void optimizeModel(IThresholds& thresholds, const IRuleInduction& ruleInduction, IPartition& partition,
                           ILabelSampling& labelSampling, IInstanceSampling& instanceSampling,
                           IFeatureSampling& featureSampling, const IRulePruning& rulePruning,
                           const IPostProcessor& postProcessor, RNG& rng) const override {
            for (auto it = postOptimizationPhases_.cbegin(); it != postOptimizationPhases_.cend(); it++) {
                const std::unique_ptr<IPostOptimizationPhase>& postOptimizationPhasePtr = *it;
                postOptimizationPhasePtr->optimizeModel(thresholds, ruleInduction, partition, labelSampling,
                                                        instanceSampling, featureSampling, rulePruning, postProcessor,
                                                        rng);
            }
        }
};

/**
 * An implementation of the class `IPostOptimization` that does not perform any optimizations, but retains a previously
 * learned rule-based model.
 */
class NoPostOptimization final : public IPostOptimization {
    private:

        const std::unique_ptr<IModelBuilder> modelBuilderPtr_;

    public:

        /**
         * @param modelBuilderPtr An unique pointer to an object of type `IModelBuilder` that should be used to build
         *                        the model
         */
        NoPostOptimization(std::unique_ptr<IModelBuilder> modelBuilderPtr)
            : modelBuilderPtr_(std::move(modelBuilderPtr)) {}

        IModelBuilder& getModelBuilder() const override {
            return *modelBuilderPtr_;
        }

        void optimizeModel(IThresholds& thresholds, const IRuleInduction& ruleInduction, IPartition& partition,
                           ILabelSampling& labelSampling, IInstanceSampling& instanceSampling,
                           IFeatureSampling& featureSampling, const IRulePruning& rulePruning,
                           const IPostProcessor& postProcessor, RNG& rng) const override {
            return;
        }
};

void PostOptimizationPhaseListFactory::addPostOptimizationPhaseFactory(
  std::unique_ptr<IPostOptimizationPhaseFactory> postOptimizationPhaseFactoryPtr) {
    postOptimizationPhaseFactories_.push_back(std::move(postOptimizationPhaseFactoryPtr));
}

std::unique_ptr<IPostOptimization> PostOptimizationPhaseListFactory::create(
  const IModelBuilderFactory& modelBuilderFactory) const {
    std::unique_ptr<IModelBuilder> modelBuilderPtr = modelBuilderFactory.create();

    if (postOptimizationPhaseFactories_.empty()) {
        return std::make_unique<NoPostOptimization>(std::move(modelBuilderPtr));
    } else {
        return std::make_unique<PostOptimizationPhaseList>(std::move(modelBuilderPtr), postOptimizationPhaseFactories_);
    }
}
