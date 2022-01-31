/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/nominal_feature_mask.hpp"
#include "common/input/feature_matrix_column_wise.hpp"
#include "common/input/label_matrix_row_wise.hpp"
#include "common/model/model_builder.hpp"
#include "common/rule_induction/rule_induction.hpp"
#include "common/sampling/label_sampling.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/sampling/feature_sampling.hpp"
#include "common/sampling/partition_sampling.hpp"
#include "common/statistics/statistics_provider.hpp"
#include "common/stopping/stopping_criterion.hpp"
#include "common/thresholds/thresholds.hpp"
#include <forward_list>


/**
 * Defines an interface for all classes that implement an algorithm for the induction of several rules that will be
 * added to a rule-based model.
 */
class IRuleModelAssemblage {

    public:

        virtual ~IRuleModelAssemblage() { };

        /**
         * Assembles and returns a rule-based model that consists of several rules.
         *
         * @param nominalFeatureMask    A reference to an object of type `INominalFeatureMask` that provides access to
         *                              the information whether individual features are nominal or not
         * @param featureMatrix         A reference to an object of type `IColumnWiseFeatureMatrix` that provides
         *                              column-wise access to the feature values of individual training examples
         * @param labelMatrix           A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise
         *                              access to the labels of individual training examples
         * @param randomState           The seed to be used by the random number generators
         * @param modelBuilder          A reference to an object of type `IModelBuilder`, the induced rules should be
         *                              added to
         * @return                      An unique pointer to an object of type `IRuleModel` that consists of the rules
         *                              that have been induced
         */
        virtual std::unique_ptr<IRuleModel> induceRules(const INominalFeatureMask& nominalFeatureMask,
                                                        const IColumnWiseFeatureMatrix& featureMatrix,
                                                        const IRowWiseLabelMatrix& labelMatrix, uint32 randomState,
                                                        IModelBuilder& modelBuilder) const = 0;

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
         * @param stoppingCriterionFactories    A reference to a `std::forward_list` that stores unique pointers to
         *                                      objects of type `IStoppingCriterionFactory` that allow to create the
         *                                      implementations to be used to decide whether additional rules should be
         *                                      induced or not
         */
        virtual std::unique_ptr<IRuleModelAssemblage> create(
            std::unique_ptr<IStatisticsProviderFactory> statisticsProviderFactoryPtr,
            std::unique_ptr<IThresholdsFactory> thresholdsFactoryPtr,
            std::unique_ptr<IRuleInductionFactory> ruleInductionFactoryPtr,
            std::unique_ptr<ILabelSamplingFactory> labelSamplingFactoryPtr,
            std::unique_ptr<IInstanceSamplingFactory> instanceSamplingFactoryPtr,
            std::unique_ptr<IFeatureSamplingFactory> featureSamplingFactoryPtr,
            std::unique_ptr<IPartitionSamplingFactory> partitionSamplingFactoryPtr,
            std::unique_ptr<IPruningFactory> pruningFactoryPtr,
            std::unique_ptr<IPostProcessorFactory> postProcessorFactoryPtr,
            std::forward_list<std::unique_ptr<IStoppingCriterionFactory>>& stoppingCriterionFactories) const = 0;

};

/**
 * Defines an interface for all classes that allow to configure an algorithm for the induction of several rules that
 * will be added to a rule-based model.
 */
class IRuleModelAssemblageConfig {

    public:

        virtual ~IRuleModelAssemblageConfig() { };

        /**
         * Creates and returns a new object of type `IRuleModelAssemblageFactory` according to specified configuration.
         *
         * @return An unique pointer to an object of type `IRuleModelAssemblageFactory` that has been created
         */
        virtual std::unique_ptr<IRuleModelAssemblageFactory> createRuleModelAssemblageFactory() const = 0;

};
