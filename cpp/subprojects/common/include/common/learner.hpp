/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/binning/feature_binning_equal_frequency.hpp"
#include "common/binning/feature_binning_equal_width.hpp"
#include "common/binning/feature_binning_no.hpp"
#include "common/input/feature_info.hpp"
#include "common/input/feature_matrix_column_wise.hpp"
#include "common/input/feature_matrix_row_wise.hpp"
#include "common/input/label_matrix_row_wise.hpp"
#include "common/multi_threading/multi_threading_manual.hpp"
#include "common/multi_threading/multi_threading_no.hpp"
#include "common/post_optimization/post_optimization_phase_list.hpp"
#include "common/post_optimization/post_optimization_sequential.hpp"
#include "common/post_optimization/post_optimization_unused_rule_removal.hpp"
#include "common/post_processing/post_processor_no.hpp"
#include "common/prediction/label_space_info.hpp"
#include "common/prediction/prediction_matrix_dense.hpp"
#include "common/prediction/prediction_matrix_sparse_binary.hpp"
#include "common/prediction/predictor_binary.hpp"
#include "common/prediction/predictor_probability.hpp"
#include "common/prediction/predictor_score.hpp"
#include "common/prediction/probability_calibration_joint.hpp"
#include "common/prediction/probability_calibration_no.hpp"
#include "common/rule_induction/rule_induction_top_down_beam_search.hpp"
#include "common/rule_induction/rule_induction_top_down_greedy.hpp"
#include "common/rule_model_assemblage/default_rule.hpp"
#include "common/rule_model_assemblage/rule_model_assemblage.hpp"
#include "common/rule_model_assemblage/rule_model_assemblage_sequential.hpp"
#include "common/rule_pruning/rule_pruning_irep.hpp"
#include "common/rule_pruning/rule_pruning_no.hpp"
#include "common/sampling/feature_sampling_no.hpp"
#include "common/sampling/feature_sampling_without_replacement.hpp"
#include "common/sampling/instance_sampling_no.hpp"
#include "common/sampling/instance_sampling_stratified_example_wise.hpp"
#include "common/sampling/instance_sampling_stratified_label_wise.hpp"
#include "common/sampling/instance_sampling_with_replacement.hpp"
#include "common/sampling/instance_sampling_without_replacement.hpp"
#include "common/sampling/label_sampling_no.hpp"
#include "common/sampling/label_sampling_round_robin.hpp"
#include "common/sampling/label_sampling_without_replacement.hpp"
#include "common/sampling/partition_sampling_bi_random.hpp"
#include "common/sampling/partition_sampling_bi_stratified_example_wise.hpp"
#include "common/sampling/partition_sampling_bi_stratified_label_wise.hpp"
#include "common/sampling/partition_sampling_no.hpp"
#include "common/stopping/global_pruning_post.hpp"
#include "common/stopping/global_pruning_pre.hpp"
#include "common/stopping/stopping_criterion_list.hpp"
#include "common/stopping/stopping_criterion_size.hpp"
#include "common/stopping/stopping_criterion_time.hpp"

/**
 * Defines an interface for all classes that provide access to the results of fitting a rule learner to training data.
 * It incorporates the model that has been trained, as well as additional information that is necessary for obtaining
 * predictions for unseen data.
 */
class MLRLCOMMON_API ITrainingResult {
    public:

        virtual ~ITrainingResult() {};

        /**
         * Returns the number of labels for which a model has been trained.
         *
         * @return The number of labels
         */
        virtual uint32 getNumLabels() const = 0;

        /**
         * Returns the model that has been trained.
         *
         * @return An unique pointer to an object of type `IRuleModel` that has been trained
         */
        virtual std::unique_ptr<IRuleModel>& getRuleModel() = 0;

        /**
         * Returns the model that has been trained.
         *
         * @return An unique pointer to an object of type `IRuleModel` that has been trained
         */
        virtual const std::unique_ptr<IRuleModel>& getRuleModel() const = 0;

        /**
         * Returns information about the label space that may be used as a basis for making predictions.
         *
         * @return An unique pointer to an object of type `ILabelSpaceInfo` that may be used as a basis for making
         *         predictions
         */
        virtual std::unique_ptr<ILabelSpaceInfo>& getLabelSpaceInfo() = 0;

        /**
         * Returns information about the label space that may be used as a basis for making predictions.
         *
         * @return An unique pointer to an object of type `ILabelSpaceInfo` that may be used as a basis for making
         *         predictions
         */
        virtual const std::unique_ptr<ILabelSpaceInfo>& getLabelSpaceInfo() const = 0;

        /**
         * Returns a model that may be used for the calibration of marginal probabilities.
         *
         * @return An unique pointer to an object of type `IMarginalProbabilityCalibrationModel` that may be used for
         *         the calibration of marginal probabilities
         */
        virtual std::unique_ptr<IMarginalProbabilityCalibrationModel>& getMarginalProbabilityCalibrationModel() = 0;

        /**
         * Returns a model that may be used for the calibration of marginal probabilities.
         *
         * @return An unique pointer to an object of type `IMarginalProbabilityCalibrationModel` that may be used for
         *         the calibration of marginal probabilities
         */
        virtual const std::unique_ptr<IMarginalProbabilityCalibrationModel>& getMarginalProbabilityCalibrationModel()
          const = 0;

        /**
         * Returns a model that may be used for the calibration of joint probabilities.
         *
         * @return An unique pointer to an object of type `IJointProbabilityCalibrationModel` that may be used for the
         *         calibration of joint probabilities
         */
        virtual std::unique_ptr<IJointProbabilityCalibrationModel>& getJointProbabilityCalibrationModel() = 0;

        /**
         * Returns a model that may be used for the calibration of joint probabilities.
         *
         * @return An unique pointer to an object of type `IJointProbabilityCalibrationModel` that may be used for the
         *         calibration of joint probabilities
         */
        virtual const std::unique_ptr<IJointProbabilityCalibrationModel>& getJointProbabilityCalibrationModel()
          const = 0;
};

/**
 * Defines an interface for all rule learners.
 */
class MLRLCOMMON_API IRuleLearner {
    public:

        /**
         * Defines an interface for all classes that allow to configure a rule learner.
         */
        class IConfig {
                friend class AbstractRuleLearner;

            protected:

                /**
                 * Returns the definition of the function that should be used for comparing the quality of different
                 * rules.
                 *
                 * @return An object of type `RuleCompareFunction` that defines the function that should be used for
                 *         comparing the quality of different rules
                 */
                virtual RuleCompareFunction getRuleCompareFunction() const = 0;

                /**
                 * Returns an unique pointer to the configuration of the default that is included in a rule-based model.
                 *
                 * @return A reference to an unique pointer of type `IDefaultRuleConfig` that stores the configuration
                 *         of the default rule that is included in a rule-based model
                 */
                virtual std::unique_ptr<IDefaultRuleConfig>& getDefaultRuleConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the algorithm for the induction of several rules
                 * that are added to a rule-based model.
                 *
                 * @return A reference to an unique pointer of type `IRuleModelAssemblageConfig` that stores the
                 *         configuration of the algorithm for the induction of several rules that are added to a
                 *         rule-based model
                 */
                virtual std::unique_ptr<IRuleModelAssemblageConfig>& getRuleModelAssemblageConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the algorithm for the induction of individual
                 * rules.
                 *
                 * @return A reference to an unique pointer of type `IRuleInductionConfig` that stores the configuration
                 *         of the algorithm for the induction of individual rules
                 */
                virtual std::unique_ptr<IRuleInductionConfig>& getRuleInductionConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the method for the assignment of numerical feature
                 * values to bins.
                 *
                 * @return A reference to an unique pointer of type `IFeatureBinningConfig` that stores the
                 *         configuration of the method for the assignment of numerical feature values to bins
                 */
                virtual std::unique_ptr<IFeatureBinningConfig>& getFeatureBinningConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the method for sampling labels.
                 *
                 * @return A reference to an unique pointer of type `ILabelSamplingConfig` that stores the configuration
                 *         of the method for sampling labels
                 */
                virtual std::unique_ptr<ILabelSamplingConfig>& getLabelSamplingConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the method for sampling instances.
                 *
                 * @return A reference to an unique pointer of type `IInstanceSamplingConfig` that stores the
                 *         configuration of the method for sampling instances
                 */
                virtual std::unique_ptr<IInstanceSamplingConfig>& getInstanceSamplingConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the method for sampling features.
                 *
                 * @return A reference to an unique pointer of type `IFeatureSamplingConfig` that specifies the
                 *         configuration of the method for sampling features
                 */
                virtual std::unique_ptr<IFeatureSamplingConfig>& getFeatureSamplingConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the method for partitioning the available training
                 * examples into a training set and a holdout set.
                 *
                 * @return A reference to an unique pointer of type `IPartitionSamplingConfig` that stores the
                 *         configuration of the method for partitioning the available training examples into a training
                 *         set and a holdout set
                 */
                virtual std::unique_ptr<IPartitionSamplingConfig>& getPartitionSamplingConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the method for pruning individual rules.
                 *
                 * @return A reference to an unique pointer of type `IRulePruningConfig` that stores the configuration
                 *         of the method for pruning individual rules
                 */
                virtual std::unique_ptr<IRulePruningConfig>& getRulePruningConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the method for post-processing the predictions of
                 * rules once they have been learned.
                 *
                 * @return A reference to an unique pointer of type `IPostProcessorConfig` that stores the configuration
                 *         of the method that post-processes the predictions of rules once they have been learned
                 */
                virtual std::unique_ptr<IPostProcessorConfig>& getPostProcessorConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the multi-threading behavior that is used for the
                 * parallel refinement of rules.
                 *
                 * @return A reference to an unique pointer of type `IMultiThreadingConfig` that stores the
                 *         configuration of the multi-threading behavior that is used for the parallel refinement of
                 *         rules
                 */
                virtual std::unique_ptr<IMultiThreadingConfig>& getParallelRuleRefinementConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the the configuration of the multi-threading behavior that is used for
                 * the parallel update of statistics.
                 *
                 * @return A reference to an unique pointer of type `IMultiThreadingConfig` that stores the
                 *         configuration of the multi-threading behavior that is used for the parallel update of
                 *         statistics
                 */
                virtual std::unique_ptr<IMultiThreadingConfig>& getParallelStatisticUpdateConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the multi-threading behavior that is used to
                 * predict for several query examples in parallel.
                 *
                 * @return A reference to an unique pointer of type `IMultiThreadingConfig` that stores the
                 *         configuration of the multi-threading behavior that is used to predict for several query
                 *         examples in parallel
                 */
                virtual std::unique_ptr<IMultiThreadingConfig>& getParallelPredictionConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the stopping criterion that ensures that the number
                 * of rules does not exceed a certain maximum.
                 *
                 * @return A reference to an unique pointer of type `SizeStoppingCriterionConfig` that stores the
                 *         configuration of the stopping criterion that ensures that the number of rules does not exceed
                 *         a certain maximum or a null pointer, if no such stopping criterion should be used
                 */
                virtual std::unique_ptr<SizeStoppingCriterionConfig>& getSizeStoppingCriterionConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the stopping criterion that ensures that a certain
                 * time limit is not exceeded.
                 *
                 * @return A reference to an unique pointer of type `TimeStoppingCriterionConfig` that stores the
                 *         configuration of the stopping criterion that ensures that a certain time limit is not
                 *         exceeded or a null pointer, if no such stopping criterion should be used
                 */
                virtual std::unique_ptr<TimeStoppingCriterionConfig>& getTimeStoppingCriterionConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the stopping criterion that allows to decide how
                 * many rules should be included in a model, such that its performance is optimized globally.
                 *
                 * @return A reference to an unique pointer of type `IGlobalPruningConfig` that stores the configuration
                 *         of the stopping criterion that allows to decide how many rules should be included in a model,
                 *         such that its performance is optimized globally, or a null pointer, if no such stopping
                 *         criterion should be used
                 */
                virtual std::unique_ptr<IGlobalPruningConfig>& getGlobalPruningConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the post-optimization method that optimizes each
                 * rule in a model by relearning it in the context of the other rules.
                 *
                 * @return A reference to an unique pointer of type `SequentialPostOptimizationConfig` that stores the
                 *         configuration of the post-optimization method that optimizes each rule in a model by
                 *         relearning it in the context of the other rules or a null pointer, if no such
                 *         post-optimization method should be used
                 */
                virtual std::unique_ptr<SequentialPostOptimizationConfig>& getSequentialPostOptimizationConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the post-optimization method that removes unused
                 * rules from a model.
                 *
                 * @return A reference to an unique pointer of type `UnusedRuleRemovalConfig` that stores the
                 *         configuration of the post-optimization method that removes unused rules from a model or a
                 *         null pointer, if no such post-optimization method should be used
                 */
                virtual std::unique_ptr<UnusedRuleRemovalConfig>& getUnusedRuleRemovalConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the calibrator that allows to fit a model for the
                 * calibration of marginal probabilities.
                 *
                 * @return A reference to an unique pointer of type `IMarginalProbabilityCalibratorConfig` that stores
                 *         the configuration of the calibrator that allows to fit a model for the calibration of
                 *         marginal probabilities
                 */
                virtual std::unique_ptr<IMarginalProbabilityCalibratorConfig>&
                  getMarginalProbabilityCalibratorConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the calibrator that allows to fit a model for the
                 * calibration of joint probabilities.
                 *
                 * @return A reference to an unique pointer of type `IJointProbabilityCalibratorConfig` that stores the
                 *         configuration of the calibrator that allows to fit a model for the calibration of joint
                 *         probabilities
                 */
                virtual std::unique_ptr<IJointProbabilityCalibratorConfig>&
                  getJointProbabilityCalibratorConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the predictor that allows to predict binary labels.
                 *
                 * @return A reference to an unique pointer of type `IBinaryPredictorConfig` that stores the
                 *         configuration of the predictor that allows to predict binary labels or a null pointer if the
                 *         prediction of binary labels is not supported
                 */
                virtual std::unique_ptr<IBinaryPredictorConfig>& getBinaryPredictorConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the predictor that allows to predict regression
                 * scores.
                 *
                 * @return A reference to an unique pointer of type `IScorePredictorConfig` that stores the
                 *         configuration of the predictor that allows to predict regression scores or a null pointer, if
                 *         the prediction of regression scores is not supported
                 */
                virtual std::unique_ptr<IScorePredictorConfig>& getScorePredictorConfigPtr() = 0;

                /**
                 * Returns an unique pointer to the configuration of the predictor that allows to predict probability
                 * estimates.
                 *
                 * @return A reference to an unique pointer of type `IProbabilityPredictorConfig` that stores the
                 *         configuration of the predictor that allows to predict probability estimates or a null
                 *         pointer, if the prediction of probability estimates is not supported
                 */
                virtual std::unique_ptr<IProbabilityPredictorConfig>& getProbabilityPredictorConfigPtr() = 0;

            public:

                virtual ~IConfig() {};
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use an algorithm that
         * sequentially induces several rules.
         */
        class ISequentialRuleModelAssemblageMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~ISequentialRuleModelAssemblageMixin() override {};

                /**
                 * Configures the rule learner to use an algorithm that sequentially induces several rules, optionally
                 * starting with a default rule, that are added to a rule-based model.
                 */
                virtual void useSequentialRuleModelAssemblage() {
                    std::unique_ptr<IRuleModelAssemblageConfig>& ruleModelAssemblageConfigPtr =
                      this->getRuleModelAssemblageConfigPtr();
                    ruleModelAssemblageConfigPtr =
                      std::make_unique<SequentialRuleModelAssemblageConfig>(this->getDefaultRuleConfigPtr());
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to induce a default rule.
         */
        class IDefaultRuleMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IDefaultRuleMixin() override {};

                /**
                 * Configures the rule learner to induce a default rule.
                 */
                virtual void useDefaultRule() {
                    std::unique_ptr<IDefaultRuleConfig>& defaultRuleConfigPtr = this->getDefaultRuleConfigPtr();
                    defaultRuleConfigPtr = std::make_unique<DefaultRuleConfig>(true);
                };
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use a greedy top-down search
         * for the induction of individual rules.
         */
        class IGreedyTopDownRuleInductionMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IGreedyTopDownRuleInductionMixin() override {};

                /**
                 * Configures the rule learner to use a greedy top-down search for the induction of individual rules.
                 *
                 * @return A reference to an object of type `IGreedyTopDownRuleInductionConfig` that allows further
                 *         configuration of the algorithm for the induction of individual rules
                 */
                virtual IGreedyTopDownRuleInductionConfig& useGreedyTopDownRuleInduction() {
                    std::unique_ptr<IRuleInductionConfig>& ruleInductionConfigPtr = this->getRuleInductionConfigPtr();
                    std::unique_ptr<GreedyTopDownRuleInductionConfig> ptr =
                      std::make_unique<GreedyTopDownRuleInductionConfig>(this->getRuleCompareFunction(),
                                                                         this->getParallelRuleRefinementConfigPtr());
                    IGreedyTopDownRuleInductionConfig& ref = *ptr;
                    ruleInductionConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use a top-down beam search.
         */
        class IBeamSearchTopDownRuleInductionMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IBeamSearchTopDownRuleInductionMixin() override {};

                /**
                 * Configures the rule learner to use a top-down beam search for the induction of individual rules.
                 *
                 * @return A reference to an object of type `IBeamSearchTopDownRuleInduction` that allows further
                 *         configuration of the algorithm for the induction of individual rules
                 */
                virtual IBeamSearchTopDownRuleInductionConfig& useBeamSearchTopDownRuleInduction() {
                    std::unique_ptr<IRuleInductionConfig>& ruleInductionConfigPtr = this->getRuleInductionConfigPtr();
                    std::unique_ptr<BeamSearchTopDownRuleInductionConfig> ptr =
                      std::make_unique<BeamSearchTopDownRuleInductionConfig>(
                        this->getRuleCompareFunction(), this->getParallelRuleRefinementConfigPtr());
                    IBeamSearchTopDownRuleInductionConfig& ref = *ptr;
                    ruleInductionConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not use any post processor.
         */
        class INoPostProcessorMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoPostProcessorMixin() override {};

                /**
                 * Configures the rule learner to not use any post processor.
                 */
                virtual void useNoPostProcessor() {
                    std::unique_ptr<IPostProcessorConfig>& postProcessorConfigPtr = this->getPostProcessorConfigPtr();
                    postProcessorConfigPtr = std::make_unique<NoPostProcessorConfig>();
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not use any method for the
         * assignment of numerical features values to bins.
         */
        class INoFeatureBinningMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoFeatureBinningMixin() override {};

                /**
                 * Configures the rule learner to not use any method for the assignment of numerical feature values to
                 * bins.
                 */
                virtual void useNoFeatureBinning() {
                    std::unique_ptr<IFeatureBinningConfig>& featureBinningConfigPtr =
                      this->getFeatureBinningConfigPtr();
                    featureBinningConfigPtr =
                      std::make_unique<NoFeatureBinningConfig>(this->getParallelStatisticUpdateConfigPtr());
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use equal-width feature
         * binning.
         */
        class IEqualWidthFeatureBinningMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IEqualWidthFeatureBinningMixin() override {};

                /**
                 * Configures the rule learner to use a method for the assignment of numerical feature values to bins,
                 * such that each bin contains values from equally sized value ranges.
                 *
                 * @return A reference to an object of type `IEqualWidthFeatureBinningConfig` that allows further
                 *         configuration of the method for the assignment of numerical feature values to bins
                 */
                virtual IEqualWidthFeatureBinningConfig& useEqualWidthFeatureBinning() {
                    std::unique_ptr<IFeatureBinningConfig>& featureBinningConfigPtr =
                      this->getFeatureBinningConfigPtr();
                    std::unique_ptr<EqualWidthFeatureBinningConfig> ptr =
                      std::make_unique<EqualWidthFeatureBinningConfig>(this->getParallelStatisticUpdateConfigPtr());
                    IEqualWidthFeatureBinningConfig& ref = *ptr;
                    featureBinningConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use equal-frequency feature
         * binning.
         */
        class IEqualFrequencyFeatureBinningMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IEqualFrequencyFeatureBinningMixin() override {};

                /**
                 * Configures the rule learner to use a method for the assignment of numerical feature values to bins,
                 * such that each bin contains approximately the same number of values.
                 *
                 * @return A reference to an object of type `IEqualFrequencyFeatureBinningConfig` that allows further
                 *         configuration of the method for the assignment of numerical feature values to bins
                 */
                virtual IEqualFrequencyFeatureBinningConfig& useEqualFrequencyFeatureBinning() {
                    std::unique_ptr<IFeatureBinningConfig>& featureBinningConfigPtr =
                      this->getFeatureBinningConfigPtr();
                    std::unique_ptr<EqualFrequencyFeatureBinningConfig> ptr =
                      std::make_unique<EqualFrequencyFeatureBinningConfig>(this->getParallelStatisticUpdateConfigPtr());
                    IEqualFrequencyFeatureBinningConfig& ref = *ptr;
                    featureBinningConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not use label sampling.
         */
        class INoLabelSamplingMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoLabelSamplingMixin() override {};

                /**
                 * Configures the rule learner to not sample from the available labels whenever a new rule should be
                 * learned.
                 */
                virtual void useNoLabelSampling() {
                    std::unique_ptr<ILabelSamplingConfig>& labelSamplingConfigPtr = this->getLabelSamplingConfigPtr();
                    labelSamplingConfigPtr = std::make_unique<NoLabelSamplingConfig>();
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use label sampling without
         * replacement.
         */
        class ILabelSamplingWithoutReplacementMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~ILabelSamplingWithoutReplacementMixin() override {};

                /**
                 * Configures the rule learner to sample from the available labels with replacement whenever a new rule
                 * should be learned.
                 *
                 * @return A reference to an object of type `ILabelSamplingWithoutReplacementConfig` that allows further
                 *         configuration of the method for sampling labels
                 */
                virtual ILabelSamplingWithoutReplacementConfig& useLabelSamplingWithoutReplacement() {
                    std::unique_ptr<ILabelSamplingConfig>& labelSamplingConfigPtr = this->getLabelSamplingConfigPtr();
                    std::unique_ptr<LabelSamplingWithoutReplacementConfig> ptr =
                      std::make_unique<LabelSamplingWithoutReplacementConfig>();
                    ILabelSamplingWithoutReplacementConfig& ref = *ptr;
                    labelSamplingConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to sample single labels in a
         * round-robin fashion.
         */
        class IRoundRobinLabelSamplingMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IRoundRobinLabelSamplingMixin() override {};

                /**
                 * Configures the rule learner to sample a single labels in a round-robin fashion whenever a new rule
                 * should be learned.
                 */
                virtual void useRoundRobinLabelSampling() {
                    std::unique_ptr<ILabelSamplingConfig>& labelSamplingConfigPtr = this->getLabelSamplingConfigPtr();
                    labelSamplingConfigPtr = std::make_unique<RoundRobinLabelSamplingConfig>();
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not use instance sampling.
         */
        class INoInstanceSamplingMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoInstanceSamplingMixin() override {};

                /**
                 * Configures the rule learner to not sample from the available training examples whenever a new rule
                 * should be learned.
                 */
                virtual void useNoInstanceSampling() {
                    std::unique_ptr<IInstanceSamplingConfig>& instanceSamplingConfigPtr =
                      this->getInstanceSamplingConfigPtr();
                    instanceSamplingConfigPtr = std::make_unique<NoInstanceSamplingConfig>();
                };
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use instance sampling with
         * replacement.
         */
        class IInstanceSamplingWithReplacementMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IInstanceSamplingWithReplacementMixin() override {};

                /**
                 * Configures the rule learner to sample from the available training examples with replacement whenever
                 * a new rule should be learned.
                 *
                 * @return A reference to an object of type `IInstanceSamplingWithReplacementConfig` that allows further
                 *         configuration of the method for sampling instances
                 */
                virtual IInstanceSamplingWithReplacementConfig& useInstanceSamplingWithReplacement() {
                    std::unique_ptr<IInstanceSamplingConfig>& instanceSamplingConfigPtr =
                      this->getInstanceSamplingConfigPtr();
                    std::unique_ptr<InstanceSamplingWithReplacementConfig> ptr =
                      std::make_unique<InstanceSamplingWithReplacementConfig>();
                    IInstanceSamplingWithReplacementConfig& ref = *ptr;
                    instanceSamplingConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use instance sampling without
         * replacement.
         */
        class IInstanceSamplingWithoutReplacementMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IInstanceSamplingWithoutReplacementMixin() override {};

                /**
                 * Configures the rule learner to sample from the available training examples without replacement
                 * whenever a new rule should be learned.
                 *
                 * @return A reference to an object of type `IInstanceSamplingWithoutReplacementConfig` that allows
                 *         further configuration of the method for sampling instances
                 */
                virtual IInstanceSamplingWithoutReplacementConfig& useInstanceSamplingWithoutReplacement() {
                    std::unique_ptr<IInstanceSamplingConfig>& instanceSamplingConfigPtr =
                      this->getInstanceSamplingConfigPtr();
                    std::unique_ptr<InstanceSamplingWithoutReplacementConfig> ptr =
                      std::make_unique<InstanceSamplingWithoutReplacementConfig>();
                    IInstanceSamplingWithoutReplacementConfig& ref = *ptr;
                    instanceSamplingConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use label-wise stratified
         * instance sampling.
         */
        class ILabelWiseStratifiedInstanceSamplingMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~ILabelWiseStratifiedInstanceSamplingMixin() override {};

                /**
                 * Configures the rule learner to sample from the available training examples using stratification, such
                 * that for each label the proportion of relevant and irrelevant examples is maintained, whenever a new
                 * rule should be learned.
                 *
                 * @return A reference to an object of type `ILabelWiseStratifiedInstanceSamplingConfig` that allows
                 *         further configuration of the method for sampling instances
                 */
                virtual ILabelWiseStratifiedInstanceSamplingConfig& useLabelWiseStratifiedInstanceSampling() {
                    std::unique_ptr<IInstanceSamplingConfig>& instanceSamplingConfigPtr =
                      this->getInstanceSamplingConfigPtr();
                    std::unique_ptr<LabelWiseStratifiedInstanceSamplingConfig> ptr =
                      std::make_unique<LabelWiseStratifiedInstanceSamplingConfig>();
                    ILabelWiseStratifiedInstanceSamplingConfig& ref = *ptr;
                    instanceSamplingConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use example-wise stratified
         * instance sampling.
         */
        class IExampleWiseStratifiedInstanceSamplingMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IExampleWiseStratifiedInstanceSamplingMixin() override {};

                /**
                 * Configures the rule learner to sample from the available training examples using stratification,
                 * where distinct label vectors are treated as individual classes, whenever a new rule should be
                 * learned.
                 *
                 * @return A reference to an object of type `IExampleWiseStratifiedInstanceSamplingConfig` that allows
                 *         further configuration of the method for sampling instances
                 */
                virtual IExampleWiseStratifiedInstanceSamplingConfig& useExampleWiseStratifiedInstanceSampling() {
                    std::unique_ptr<IInstanceSamplingConfig>& instanceSamplingConfigPtr =
                      this->getInstanceSamplingConfigPtr();
                    std::unique_ptr<ExampleWiseStratifiedInstanceSamplingConfig> ptr =
                      std::make_unique<ExampleWiseStratifiedInstanceSamplingConfig>();
                    IExampleWiseStratifiedInstanceSamplingConfig& ref = *ptr;
                    instanceSamplingConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not use feature sampling.
         */
        class INoFeatureSamplingMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoFeatureSamplingMixin() override {};

                /**
                 * Configures the rule learner to not sample from the available features whenever a rule should be
                 * refined.
                 */
                virtual void useNoFeatureSampling() {
                    std::unique_ptr<IFeatureSamplingConfig>& featureSamplingConfigPtr =
                      this->getFeatureSamplingConfigPtr();
                    featureSamplingConfigPtr = std::make_unique<NoFeatureSamplingConfig>();
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use feature sampling without
         * replacement.
         */
        class IFeatureSamplingWithoutReplacementMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IFeatureSamplingWithoutReplacementMixin() override {};

                /**
                 * Configures the rule learner to sample from the available features with replacement whenever a rule
                 * should be refined.
                 *
                 * @return A reference to an object of type `IFeatureSamplingWithoutReplacementConfig` that allows
                 *         further configuration of the method for sampling features
                 */
                virtual IFeatureSamplingWithoutReplacementConfig& useFeatureSamplingWithoutReplacement() {
                    std::unique_ptr<IFeatureSamplingConfig>& featureSamplingConfigPtr =
                      this->getFeatureSamplingConfigPtr();
                    std::unique_ptr<FeatureSamplingWithoutReplacementConfig> ptr =
                      std::make_unique<FeatureSamplingWithoutReplacementConfig>();
                    IFeatureSamplingWithoutReplacementConfig& ref = *ptr;
                    featureSamplingConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not partition the available
         * training examples into a training set and a holdout set.
         */
        class INoPartitionSamplingMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoPartitionSamplingMixin() override {};

                /**
                 * Configures the rule learner to not partition the available training examples into a training set and
                 * a holdout set.
                 */
                virtual void useNoPartitionSampling() {
                    std::unique_ptr<IPartitionSamplingConfig>& partitionSamplingConfigPtr =
                      this->getPartitionSamplingConfigPtr();
                    partitionSamplingConfigPtr = std::make_unique<NoPartitionSamplingConfig>();
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to partition the available
         * training example into a training set and a holdout set by randomly splitting the training examples into two
         * mutually exclusive sets.
         */
        class IRandomBiPartitionSamplingMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IRandomBiPartitionSamplingMixin() override {};

                /**
                 * Configures the rule learner to partition the available training examples into a training set and a
                 * holdout set by randomly splitting the training examples into two mutually exclusive sets.
                 *
                 * @return A reference to an object of type `IRandomBiPartitionSamplingConfig` that allows further
                 *         configuration of the method for partitioning the available training examples into a training
                 *         set and a holdout set
                 */
                virtual IRandomBiPartitionSamplingConfig& useRandomBiPartitionSampling() {
                    std::unique_ptr<IPartitionSamplingConfig>& partitionSamplingConfigPtr =
                      this->getPartitionSamplingConfigPtr();
                    std::unique_ptr<RandomBiPartitionSamplingConfig> ptr =
                      std::make_unique<RandomBiPartitionSamplingConfig>();
                    IRandomBiPartitionSamplingConfig& ref = *ptr;
                    partitionSamplingConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to partition the available
         * training examples into a training set and a holdout set using stratification, such that for each label the
         * proportion of relevant and irrelevant examples is maintained.
         */
        class ILabelWiseStratifiedBiPartitionSamplingMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~ILabelWiseStratifiedBiPartitionSamplingMixin() override {};

                /**
                 * Configures the rule learner to partition the available training examples into a training set and a
                 * holdout set using stratification, such that for each label the proportion of relevant and irrelevant
                 * examples is maintained.
                 *
                 * @return A reference to an object of type `ILabelWiseStratifiedBiPartitionSamplingConfig` that allows
                 *         further configuration of the method for partitioning the available training examples into a
                 *         training and a holdout set
                 */
                virtual ILabelWiseStratifiedBiPartitionSamplingConfig& useLabelWiseStratifiedBiPartitionSampling() {
                    std::unique_ptr<IPartitionSamplingConfig>& partitionSamplingConfigPtr =
                      this->getPartitionSamplingConfigPtr();
                    std::unique_ptr<LabelWiseStratifiedBiPartitionSamplingConfig> ptr =
                      std::make_unique<LabelWiseStratifiedBiPartitionSamplingConfig>();
                    ILabelWiseStratifiedBiPartitionSamplingConfig& ref = *ptr;
                    partitionSamplingConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to partition the available
         * training examples into a training set and a holdout set using stratification, where distinct label vectors
         * are treated as individual classes.
         */
        class IExampleWiseStratifiedBiPartitionSamplingMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IExampleWiseStratifiedBiPartitionSamplingMixin() override {};

                /**
                 * Configures the rule learner to partition the available training examples into a training set and a
                 * holdout set using stratification, where distinct label vectors are treated as individual classes
                 *
                 * @return A reference to an object of type `IExampleWiseStratifiedBiPartitionSamplingConfig` that
                 *         allows further configuration of the method for partitioning the available training examples
                 *         into a training and a holdout set
                 */
                virtual IExampleWiseStratifiedBiPartitionSamplingConfig& useExampleWiseStratifiedBiPartitionSampling() {
                    std::unique_ptr<IPartitionSamplingConfig>& partitionSamplingConfigPtr =
                      this->getPartitionSamplingConfigPtr();
                    std::unique_ptr<ExampleWiseStratifiedBiPartitionSamplingConfig> ptr =
                      std::make_unique<ExampleWiseStratifiedBiPartitionSamplingConfig>();
                    IExampleWiseStratifiedBiPartitionSamplingConfig& ref = *ptr;
                    partitionSamplingConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not prune individual rules.
         */
        class INoRulePruningMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoRulePruningMixin() override {};

                /**
                 * Configures the rule learner to not prune individual rules.
                 */
                virtual void useNoRulePruning() {
                    std::unique_ptr<IRulePruningConfig>& rulePruningConfigPtr = this->getRulePruningConfigPtr();
                    rulePruningConfigPtr = std::make_unique<NoRulePruningConfig>();
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to prune individual rules by
         * following the principles of "incremental reduced error pruning" (IREP).
         */
        class IIrepRulePruningMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IIrepRulePruningMixin() override {};

                /**
                 * Configures the rule learner to prune individual rules by following the principles of "incremental
                 * reduced error pruning" (IREP).
                 */
                virtual void useIrepRulePruning() {
                    std::unique_ptr<IRulePruningConfig>& rulePruningConfigPtr = this->getRulePruningConfigPtr();
                    rulePruningConfigPtr = std::make_unique<IrepConfig>(this->getRuleCompareFunction());
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not use any multi-threading
         * for the parallel refinement of rules.
         */
        class INoParallelRuleRefinementMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoParallelRuleRefinementMixin() override {}

                /**
                 * Configures the rule learner to not use any multi-threading for the parallel refinement of rules.
                 */
                virtual void useNoParallelRuleRefinement() {
                    std::unique_ptr<IMultiThreadingConfig>& parallelRuleRefinementConfigPtr =
                      this->getParallelRuleRefinementConfigPtr();
                    parallelRuleRefinementConfigPtr = std::make_unique<NoMultiThreadingConfig>();
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use multi-threading for the
         * parallel refinement of rules.
         */
        class IParallelRuleRefinementMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IParallelRuleRefinementMixin() override {};

                /**
                 * Configures the rule learner to use multi-threading for the parallel refinement of rules.
                 *
                 * @return A reference to an object of type `IManualMultiThreadingConfig` that allows further
                 *         configuration of the multi-threading behavior
                 */
                virtual IManualMultiThreadingConfig& useParallelRuleRefinement() {
                    std::unique_ptr<IMultiThreadingConfig>& parallelRuleRefinementConfigPtr =
                      this->getParallelRuleRefinementConfigPtr();
                    std::unique_ptr<ManualMultiThreadingConfig> ptr = std::make_unique<ManualMultiThreadingConfig>();
                    IManualMultiThreadingConfig& ref = *ptr;
                    parallelRuleRefinementConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not use any multi-threading
         * for the parallel update of statistics.
         */
        class INoParallelStatisticUpdateMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoParallelStatisticUpdateMixin() override {};

                /**
                 * Configures the rule learner to not use any multi-threading for the parallel update of statistics.
                 */
                virtual void useNoParallelStatisticUpdate() {
                    std::unique_ptr<IMultiThreadingConfig>& parallelStatisticUpdateConfigPtr =
                      this->getParallelStatisticUpdateConfigPtr();
                    parallelStatisticUpdateConfigPtr = std::make_unique<NoMultiThreadingConfig>();
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use multi-threading for the
         * parallel update of statistics.
         */
        class IParallelStatisticUpdateMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IParallelStatisticUpdateMixin() override {};

                /**
                 * Configures the rule learner to use multi-threading for the parallel update of statistics.
                 *
                 * @return A reference to an object of type `IManualMultiThreadingConfig` that allows further
                 *         configuration of the multi-threading behavior
                 */
                virtual IManualMultiThreadingConfig& useParallelStatisticUpdate() {
                    std::unique_ptr<IMultiThreadingConfig>& parallelStatisticUpdateConfigPtr =
                      this->getParallelStatisticUpdateConfigPtr();
                    std::unique_ptr<ManualMultiThreadingConfig> ptr = std::make_unique<ManualMultiThreadingConfig>();
                    IManualMultiThreadingConfig& ref = *ptr;
                    parallelStatisticUpdateConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not use any multi-threading
         * for prediction.
         */
        class INoParallelPredictionMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoParallelPredictionMixin() override {};

                /**
                 * Configures the rule learner to not use any multi-threading to predict for several query examples in
                 * parallel.
                 */
                virtual void useNoParallelPrediction() {
                    std::unique_ptr<IMultiThreadingConfig>& parallelPredictionConfigPtr =
                      this->getParallelPredictionConfigPtr();
                    parallelPredictionConfigPtr = std::make_unique<NoMultiThreadingConfig>();
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use multi-threading to predict
         * for several examples in parallel.
         */
        class IParallelPredictionMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IParallelPredictionMixin() override {};

                /**
                 * Configures the rule learner to use multi-threading to predict for several query examples in parallel.
                 *
                 * @return A reference to an object of type `IManualMultiThreadingConfig` that allows further
                 *         configuration of the multi-threading behavior
                 */
                virtual IManualMultiThreadingConfig& useParallelPrediction() {
                    std::unique_ptr<IMultiThreadingConfig>& parallelPredictionConfigPtr =
                      this->getParallelPredictionConfigPtr();
                    std::unique_ptr<ManualMultiThreadingConfig> ptr = std::make_unique<ManualMultiThreadingConfig>();
                    IManualMultiThreadingConfig& ref = *ptr;
                    parallelPredictionConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not use a stopping criterion
         * that ensures that the number of induced rules does not exceed a certain maximum.
         */
        class INoSizeStoppingCriterionMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoSizeStoppingCriterionMixin() override {};

                /**
                 * Configures the rule learner to not use a stopping criterion that ensures that the number of induced
                 * rules does not exceed a certain maximum.
                 */
                virtual void useNoSizeStoppingCriterion() {
                    std::unique_ptr<SizeStoppingCriterionConfig>& sizeStoppingCriterionConfigPtr =
                      this->getSizeStoppingCriterionConfigPtr();
                    sizeStoppingCriterionConfigPtr = nullptr;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use a stopping criterion that
         * ensures that the number of induced rules does not exceed a certain maximum.
         */
        class ISizeStoppingCriterionMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~ISizeStoppingCriterionMixin() override {};

                /**
                 * Configures the rule learner to use a stopping criterion that ensures that the number of induced rules
                 * does not exceed a certain maximum.
                 *
                 * @return A reference to an object of type `ISizeStoppingCriterionConfig` that allows further
                 *         configuration of the stopping criterion
                 */
                virtual ISizeStoppingCriterionConfig& useSizeStoppingCriterion() {
                    std::unique_ptr<SizeStoppingCriterionConfig>& sizeStoppingCriterionConfigPtr =
                      this->getSizeStoppingCriterionConfigPtr();
                    std::unique_ptr<SizeStoppingCriterionConfig> ptr = std::make_unique<SizeStoppingCriterionConfig>();
                    ISizeStoppingCriterionConfig& ref = *ptr;
                    sizeStoppingCriterionConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not use a stopping criterion
         * that ensures that a certain time limit is not exceeded.
         */
        class INoTimeStoppingCriterionMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoTimeStoppingCriterionMixin() override {};

                /**
                 * Configures the rule learner to not use a stopping criterion that ensures that a certain time limit is
                 * not exceeded.
                 */
                virtual void useNoTimeStoppingCriterion() {
                    std::unique_ptr<TimeStoppingCriterionConfig>& timeStoppingCriterionConfigPtr =
                      this->getTimeStoppingCriterionConfigPtr();
                    timeStoppingCriterionConfigPtr = nullptr;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use a stopping criterion that
         * ensures that a certain time limit is not exceeded.
         */
        class ITimeStoppingCriterionMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~ITimeStoppingCriterionMixin() override {};

                /**
                 * Configures the rule learner to use a stopping criterion that ensures that a certain time limit is not
                 * exceeded.
                 *
                 * @return A reference to an object of type `ITimeStoppingCriterionConfig` that allows further
                 *         configuration of the stopping criterion
                 */
                virtual ITimeStoppingCriterionConfig& useTimeStoppingCriterion() {
                    std::unique_ptr<TimeStoppingCriterionConfig>& timeStoppingCriterionConfigPtr =
                      this->getTimeStoppingCriterionConfigPtr();
                    std::unique_ptr<TimeStoppingCriterionConfig> ptr = std::make_unique<TimeStoppingCriterionConfig>();
                    ITimeStoppingCriterionConfig& ref = *ptr;
                    timeStoppingCriterionConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use a stopping criterion that
         * stops the induction of rules as soon as the quality of a model's predictions for the examples in the training
         * or holdout set do not improve according to a certain measure.
         */
        class IPrePruningMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IPrePruningMixin() override {};

                /**
                 * Configures the rule learner to use a stopping criterion that stops the induction of rules as soon as
                 * the quality of a model's predictions for the examples in the training or holdout set do not improve
                 * according to a certain measure.
                 *
                 * @return A reference to an object of the type `IPrePruningConfig` that allows further configuration of
                 *         the stopping criterion
                 */
                virtual IPrePruningConfig& useGlobalPrePruning() {
                    std::unique_ptr<IGlobalPruningConfig>& globalPruningConfigPtr = this->getGlobalPruningConfigPtr();
                    std::unique_ptr<PrePruningConfig> ptr = std::make_unique<PrePruningConfig>();
                    IPrePruningConfig& ref = *ptr;
                    globalPruningConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not use global pruning.
         */
        class INoGlobalPruningMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoGlobalPruningMixin() override {};

                /**
                 * Configures the rule learner to not use global pruning.
                 */
                virtual void useNoGlobalPruning() {
                    std::unique_ptr<IGlobalPruningConfig>& globalPruningConfigPtr = this->getGlobalPruningConfigPtr();
                    globalPruningConfigPtr = nullptr;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use a stopping criterion that
         * keeps track of the number of rules in a model that perform best with respect to the examples in the training
         * or holdout set according to a certain measure.
         */
        class IPostPruningMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~IPostPruningMixin() override {};

                /**
                 * Configures the rule learner to use a stopping criterion that keeps track of the number of rules in a
                 * model that perform best with respect to the examples in the training or holdout set according to a
                 * certain measure.
                 */
                virtual IPostPruningConfig& useGlobalPostPruning() {
                    std::unique_ptr<IGlobalPruningConfig>& globalPruningConfigPtr = this->getGlobalPruningConfigPtr();
                    std::unique_ptr<PostPruningConfig> ptr = std::make_unique<PostPruningConfig>();
                    IPostPruningConfig& ref = *ptr;
                    globalPruningConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not use a post-optimization
         * method that optimizes each rule in a model by relearning it in the context of the other rules.
         */
        class INoSequentialPostOptimizationMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoSequentialPostOptimizationMixin() override {};

                /**
                 * Configures the rule learner to not use a post-optimization method that optimizes each rule in a model
                 * by relearning it in the context of the other rules.
                 */
                virtual void useNoSequentialPostOptimization() {
                    std::unique_ptr<SequentialPostOptimizationConfig>& sequentialPostOptimizationConfigPtr =
                      this->getSequentialPostOptimizationConfigPtr();
                    sequentialPostOptimizationConfigPtr = nullptr;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to use a post-optimization method
         * that optimizes each rule in a model by relearning it in the context of the other rules.
         */
        class ISequentialPostOptimizationMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~ISequentialPostOptimizationMixin() override {};

                /**
                 * Configures the rule learner to use a post-optimization method that optimizes each rule in a model by
                 * relearning it in the context of the other rules.
                 *
                 * @return A reference to an object of type `ISequentialPostOptimizationConfig` that allows further
                 *         configuration of the post-optimization method
                 */
                virtual ISequentialPostOptimizationConfig& useSequentialPostOptimization() {
                    std::unique_ptr<SequentialPostOptimizationConfig>& sequentialPostOptimizationConfigPtr =
                      this->getSequentialPostOptimizationConfigPtr();
                    std::unique_ptr<SequentialPostOptimizationConfig> ptr =
                      std::make_unique<SequentialPostOptimizationConfig>();
                    ISequentialPostOptimizationConfig& ref = *ptr;
                    sequentialPostOptimizationConfigPtr = std::move(ptr);
                    return ref;
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not calibrate marginal
         * probabilities.
         */
        class INoMarginalProbabilityCalibrationMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoMarginalProbabilityCalibrationMixin() override {};

                /**
                 * Configures the rule learner to not calibrate marginal probabilities.
                 */
                virtual void useNoMarginalProbabilityCalibration() {
                    std::unique_ptr<IMarginalProbabilityCalibratorConfig>& marginalProbabilityCalibratorConfigPtr =
                      this->getMarginalProbabilityCalibratorConfigPtr();
                    marginalProbabilityCalibratorConfigPtr = std::make_unique<NoMarginalProbabilityCalibratorConfig>();
                }
        };

        /**
         * Defines an interface for all classes that allow to configure a rule learner to not calibrate joint
         * probabilities.
         */
        class INoJointProbabilityCalibrationMixin : virtual public IRuleLearner::IConfig {
            public:

                virtual ~INoJointProbabilityCalibrationMixin() override {};

                /**
                 * Configures the rule learner to not calibrate joint probabilities.
                 */
                virtual void useNoJointProbabilityCalibration() {
                    std::unique_ptr<IJointProbabilityCalibratorConfig>& jointProbabilityCalibratorConfigPtr =
                      this->getJointProbabilityCalibratorConfigPtr();
                    jointProbabilityCalibratorConfigPtr = std::make_unique<NoJointProbabilityCalibratorConfig>();
                }
        };

        virtual ~IRuleLearner() {};

        /**
         * Applies the rule learner to given training examples and corresponding ground truth labels.
         *
         * @param featureInfo       A reference to an object of type `IFeatureInfo` that provides information about the
         *                          types of individual features
         * @param featureMatrix     A reference to an object of type `IColumnWiseFeatureMatrix` that provides
         *                          column-wise access to the feature values of the training examples
         * @param labelMatrix       A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access
         *                          to the ground truth labels of the training examples
         * @param randomState       The seed to be used by random number generators
         * @return                  An unique pointer to an object of type `ITrainingResult` that provides access to the
         *                          results of fitting the rule learner to the training data
         */
        virtual std::unique_ptr<ITrainingResult> fit(const IFeatureInfo& featureInfo,
                                                     const IColumnWiseFeatureMatrix& featureMatrix,
                                                     const IRowWiseLabelMatrix& labelMatrix,
                                                     uint32 randomState) const = 0;

        /**
         * Returns whether the rule learner is able to predict binary labels or not.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param trainingResult    A reference to an object of type `ITrainingResult` that provides access to the model
         *                          and additional information that should be used to obtain predictions
         * @return                  True, if the rule learner is able to predict binary labels, false otherwise
         */
        virtual bool canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix,
                                      const ITrainingResult& trainingResult) const = 0;

        /**
         * Returns whether the rule learner is able to predict binary labels or not.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param numLabels         The number of labels to predict for
         * @return                  True, if the rule learner is able to predict binary labels, false otherwise
         */
        virtual bool canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict binary labels for given query examples. If the
         * prediction of binary labels is not supported by the rule learner, a `std::runtime_error` is thrown.
         *
         * @throws std::runtime_exception   The exception that is thrown if the prediction of binary labels is not
         *                                  supported by the rule learner
         * @param featureMatrix             A reference to an object of type `IRowWiseFeatureMatrix` that provides
         *                                  row-wise access to the feature values of the query examples
         * @param trainingResult            A reference to an object of type `ITrainingResult` that provides access to
         *                                  the model and additional information that should be used to obtain
         *                                  predictions
         * @return                          An unique pointer to an object of type `IBinaryPredictor` that may be used
         *                                  to predict binary labels for the given query examples
         */
        virtual std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict binary labels for given query examples. If the
         * prediction of binary labels is not supported by the rule learner, a `std::runtime_error` is thrown.
         *
         * @throws std::runtime_exception             The exception that is thrown if the prediction of binary labels is
         *                                            not supported by the rule learner
         * @param featureMatrix                       A reference to an object of type `IRowWiseFeatureMatrix` that
         *                                            provides row-wise access to the feature values of the query
         *                                            examples
         * @param ruleModel                           A reference to an object of type `IRuleModel` that should be used
         *                                            to obtain predictions
         * @param labelSpaceInfo                      A reference to an object of type `ILabelSpaceInfo` that provides
         *                                            information about the label space that may be used as a basis for
         *                                            obtaining predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `IBinaryPredictor` that may
         *                                            be used to predict binary labels for the given query examples
         */
        virtual std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict sparse binary labels for given query examples. If
         * the prediction of sparse binary labels is not supported by the rule learner, a `std::runtime_error` is
         * thrown.
         *
         * @throws std::runtime_exception   The exception that is thrown if the prediction of sparse binary labels is
         *                                  not supported by the rule learner
         * @param featureMatrix             A reference to an object of type `IRowWiseFeatureMatrix` that provides
         *                                  row-wise access to the feature values of the query examples
         * @param trainingResult            A reference to an object of type `ITrainingResult` that provides access to
         *                                  the model and additional information that should be used to obtain
         *                                  predictions
         * @return                          An unique pointer to an object of type `ISparseBinaryPredictor` that may be
         *                                  used to predict sparse binary labels for the given query examples
         */
        virtual std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict sparse binary labels for given query examples. If
         * the prediction of sparse binary labels is not supported by the rule learner, a `std::runtime_error` is
         * thrown.
         *
         * @throws std::runtime_exception             The exception that is thrown if the prediction of sparse binary
         *                                            labels is not supported by the rule learner
         * @param featureMatrix                       A reference to an object of type `IRowWiseFeatureMatrix` that
         *                                            provides row-wise access to the feature values of the query
         *                                            examples
         * @param ruleModel                           A reference to an object of type `IRuleModel` that should be used
         *                                            to obtain predictions
         * @param labelSpaceInfo                      A reference to an object of type `ILabelSpaceInfo` that provides
         *                                            information about the label space that may be used as a basis for
         *                                            obtaining predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `ISparseBinaryPredictor`
         *                                            that may be used to predict sparse binary labels for the given
         *                                            query examples
         */
        virtual std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Returns whether the rule learner is able to predict regression scores or not.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param trainingResult    A reference to an object of type `ITrainingResult` that provides access to the model
         *                          and additional information that should be used to obtain predictions
         * @return                  True, if the rule learner is able to predict regression scores, false otherwise
         */
        virtual bool canPredictScores(const IRowWiseFeatureMatrix& featureMatrix,
                                      const ITrainingResult& trainingResult) const = 0;

        /**
         * Returns whether the rule learner is able to predict regression scores or not.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param numLabels         The number of labels to predict for
         * @return                  True, if the rule learner is able to predict regression scores, false otherwise
         */
        virtual bool canPredictScores(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict regression scores for given query examples. If
         * the prediction of regression scores is not supported by the rule learner, a `std::runtime_error` is thrown.
         *
         * @throws std::runtime_exception   The exception that is thrown if the prediction of regression scores is not
         *                                  supported by the rule learner
         * @param featureMatrix             A reference to an object of type `IRowWiseFeatureMatrix` that provides
         *                                  row-wise access to the feature values of the query examples
         * @param trainingResult            A reference to an object of type `ITrainingResult` that provides access to
         *                                  the model and additional information that should be used to obtain
         *                                  predictions
         * @return                          An unique pointer to an object of type `IScorePredictor` that may be used to
         *                                  predict regression scores for the given query examples
         */
        virtual std::unique_ptr<IScorePredictor> createScorePredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                                      const ITrainingResult& trainingResult) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict regression scores for given query examples. If
         * the prediction of regression scores is not supported by the rule learner, a `std::runtime_error` is thrown.
         *
         * @throws std::runtime_exception The exception that is thrown if the prediction of regression scores is not
         *                                supported by the rule learner
         * @param featureMatrix           A reference to an object of type `IRowWiseFeatureMatrix` that provides
         *                                row-wise access to the feature values of the query examples
         * @param ruleModel               A reference to an object of type `IRuleModel` that should be used to obtain
         *                                predictions
         * @param labelSpaceInfo          A reference to an object of type `ILabelSpaceInfo` that provides information
         *                                about the label space that may be used as a basis for obtaining predictions
         * @param numLabels               The number of labels to predict for
         * @return                        An unique pointer to an object of type `IScorePredictor` that may be used to
         *                                predict regression scores for the given query examples
         */
        virtual std::unique_ptr<IScorePredictor> createScorePredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                                      const IRuleModel& ruleModel,
                                                                      const ILabelSpaceInfo& labelSpaceInfo,
                                                                      uint32 numLabels) const = 0;

        /**
         * Returns whether the rule learner is able to predict probabilities or not.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param trainingResult    A reference to an object of type `ITrainingResult` that provides access to the model
         *                          and additional information that should be used to obtain predictions
         * @return                  True, if the rule learner is able to predict probabilities, false otherwise
         */
        virtual bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix,
                                             const ITrainingResult& trainingResult) const = 0;

        /**
         * Returns whether the rule learner is able to predict probabilities or not.
         *
         * @param featureMatrix     A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise
         *                          access to the feature values of the query examples
         * @param numLabels         The number of labels to predict for
         * @return                  True, if the rule learner is able to predict probabilities, false otherwise
         */
        virtual bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict probability estimates for given query examples.
         * If the prediction of probability estimates is not supported by the rule learner, a `std::runtime_error` is
         * thrown.
         *
         * @throws std::runtime_exception   The exception that is thrown if the prediction of probability estimates is
         *                                  not supported by the rule learner
         * @param featureMatrix             A reference to an object of type `IRowWiseFeatureMatrix` that provides
         *                                  row-wise access to the feature values of the query examples
         * @param trainingResult            A reference to an object of type `ITrainingResult` that provides access to
         *                                  the model and additional information that should be used to obtain
         *                                  predictions
         * @return                          An unique pointer to an object of type `IProbabilityPredictor` that may be
         *                                  used to predict probability estimates for the given query examples
         */
        virtual std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const = 0;

        /**
         * Creates and returns a predictor that may be used to predict probability estimates for given query examples.
         * If the prediction of probability estimates is not supported by the rule learner, a `std::runtime_error` is
         * thrown.
         *
         * @throws std::runtime_exception             The exception that is thrown if the prediction of probability
         *                                            estimates is not supported by the rule learner
         * @param featureMatrix                       A reference to an object of type `IRowWiseFeatureMatrix` that
         *                                            provides row-wise access to the feature values of the query
         *                                            examples
         * @param ruleModel                           A reference to an object of type `IRuleModel` that should be used
         *                                            to obtain predictions
         * @param labelSpaceInfo                      A reference to an object of type `ILabelSpaceInfo` that provides
         *                                            information about the label space that may be used as a basis for
         *                                            obtaining predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `IProbabilityPredictor`
         *                                            that may be used to predict probability estimates for the given
         *                                            query examples
         */
        virtual std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;
};

/**
 * An abstract base class for all rule learners.
 */
class AbstractRuleLearner : virtual public IRuleLearner {
    public:

        /**
         * Allows to configure a rule learner.
         */
        class Config : virtual public IRuleLearner::IConfig {
            private:

                const RuleCompareFunction ruleCompareFunction_;

            protected:

                /**
                 * An unique pointer that stores the configuration of the default rule that is included in a rule-based
                 * model.
                 */
                std::unique_ptr<IDefaultRuleConfig> defaultRuleConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for the induction of several rules that
                 * are added to a rule-based model.
                 */
                std::unique_ptr<IRuleModelAssemblageConfig> ruleModelAssemblageConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the algorithm for the induction of individual
                 * rules.
                 */
                std::unique_ptr<IRuleInductionConfig> ruleInductionConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for the assignment of numerical feature
                 * values to bins
                 */
                std::unique_ptr<IFeatureBinningConfig> featureBinningConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for sampling labels.
                 */
                std::unique_ptr<ILabelSamplingConfig> labelSamplingConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for sampling instances.
                 */
                std::unique_ptr<IInstanceSamplingConfig> instanceSamplingConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for sampling features.
                 */
                std::unique_ptr<IFeatureSamplingConfig> featureSamplingConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for partitioning the available training
                 * examples into a training set and a holdout set.
                 */
                std::unique_ptr<IPartitionSamplingConfig> partitionSamplingConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for pruning individual rules.
                 */
                std::unique_ptr<IRulePruningConfig> rulePruningConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the method for post-processing the predictions of
                 * rules once they have been learned.
                 */
                std::unique_ptr<IPostProcessorConfig> postProcessorConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the multi-threading behavior that is used for the
                 * parallel refinement of rules.
                 */
                std::unique_ptr<IMultiThreadingConfig> parallelRuleRefinementConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the multi-threading behavior that is used for the
                 * parallel update of statistics.
                 */
                std::unique_ptr<IMultiThreadingConfig> parallelStatisticUpdateConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the multi-threading behavior that is used to
                 * predict for several query examples in parallel.
                 */
                std::unique_ptr<IMultiThreadingConfig> parallelPredictionConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the stopping criterion that ensures that the
                 * number of rules does not exceed a certain maximum.
                 */
                std::unique_ptr<SizeStoppingCriterionConfig> sizeStoppingCriterionConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the stopping criterion that ensures that a certain
                 * time limit is not exceeded.
                 */
                std::unique_ptr<TimeStoppingCriterionConfig> timeStoppingCriterionConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the stopping criterion that allows to decide how
                 * many rules should be included in a model, such that its performance is optimized globally.
                 */
                std::unique_ptr<IGlobalPruningConfig> globalPruningConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the post-optimization method that optimizes each
                 * rule in a model by relearning it in the context of the other rules.
                 */
                std::unique_ptr<SequentialPostOptimizationConfig> sequentialPostOptimizationConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the post-optimization method that removes unused
                 * rules from a model.
                 */
                std::unique_ptr<UnusedRuleRemovalConfig> unusedRuleRemovalConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the calibrator that allows to fit a model for the
                 * calibration of marginal probabilities.
                 */
                std::unique_ptr<IMarginalProbabilityCalibratorConfig> marginalProbabilityCalibratorConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the calibrator that allows to fit a model for the
                 * calibration of joint probabilities.
                 */
                std::unique_ptr<IJointProbabilityCalibratorConfig> jointProbabilityCalibratorConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the predictor that allows to predict binary
                 * labels.
                 */
                std::unique_ptr<IBinaryPredictorConfig> binaryPredictorConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the predictor that allows to predict regression
                 * scores.
                 */
                std::unique_ptr<IScorePredictorConfig> scorePredictorConfigPtr_;

                /**
                 * An unique pointer that stores the configuration of the predictor that allows to predict probability
                 * estimates.
                 */
                std::unique_ptr<IProbabilityPredictorConfig> probabilityPredictorConfigPtr_;

            private:

                RuleCompareFunction getRuleCompareFunction() const override final;

                std::unique_ptr<IDefaultRuleConfig>& getDefaultRuleConfigPtr() override final;

                std::unique_ptr<IRuleModelAssemblageConfig>& getRuleModelAssemblageConfigPtr() override final;

                std::unique_ptr<IRuleInductionConfig>& getRuleInductionConfigPtr() override final;

                std::unique_ptr<IFeatureBinningConfig>& getFeatureBinningConfigPtr() override final;

                std::unique_ptr<ILabelSamplingConfig>& getLabelSamplingConfigPtr() override final;

                std::unique_ptr<IInstanceSamplingConfig>& getInstanceSamplingConfigPtr() override final;

                std::unique_ptr<IFeatureSamplingConfig>& getFeatureSamplingConfigPtr() override final;

                std::unique_ptr<IPartitionSamplingConfig>& getPartitionSamplingConfigPtr() override final;

                std::unique_ptr<IRulePruningConfig>& getRulePruningConfigPtr() override final;

                std::unique_ptr<IPostProcessorConfig>& getPostProcessorConfigPtr() override final;

                std::unique_ptr<IMultiThreadingConfig>& getParallelRuleRefinementConfigPtr() override final;

                std::unique_ptr<IMultiThreadingConfig>& getParallelStatisticUpdateConfigPtr() override final;

                std::unique_ptr<IMultiThreadingConfig>& getParallelPredictionConfigPtr() override final;

                std::unique_ptr<SizeStoppingCriterionConfig>& getSizeStoppingCriterionConfigPtr() override final;

                std::unique_ptr<TimeStoppingCriterionConfig>& getTimeStoppingCriterionConfigPtr() override final;

                std::unique_ptr<IGlobalPruningConfig>& getGlobalPruningConfigPtr() override final;

                std::unique_ptr<SequentialPostOptimizationConfig>& getSequentialPostOptimizationConfigPtr()
                  override final;

                std::unique_ptr<UnusedRuleRemovalConfig>& getUnusedRuleRemovalConfigPtr() override final;

                std::unique_ptr<IMarginalProbabilityCalibratorConfig>& getMarginalProbabilityCalibratorConfigPtr()
                  override final;

                std::unique_ptr<IJointProbabilityCalibratorConfig>& getJointProbabilityCalibratorConfigPtr()
                  override final;

                std::unique_ptr<IBinaryPredictorConfig>& getBinaryPredictorConfigPtr() override final;

                std::unique_ptr<IScorePredictorConfig>& getScorePredictorConfigPtr() override final;

                std::unique_ptr<IProbabilityPredictorConfig>& getProbabilityPredictorConfigPtr() override final;

            public:

                /**
                 * @param ruleCompareFunction An object of type `RuleCompareFunction` that defines the function that
                 *                            should be used for comparing the quality of different rules
                 */
                Config(RuleCompareFunction ruleCompareFunction);
        };

    private:

        IRuleLearner::IConfig& config_;

        std::unique_ptr<IRuleModelAssemblageFactory> createRuleModelAssemblageFactory(
          const IRowWiseLabelMatrix& labelMatrix) const;

        std::unique_ptr<IThresholdsFactory> createThresholdsFactory(const IFeatureMatrix& featureMatrix,
                                                                    const ILabelMatrix& labelMatrix) const;

        std::unique_ptr<IRuleInductionFactory> createRuleInductionFactory(const IFeatureMatrix& featureMatrix,
                                                                          const ILabelMatrix& labelMatrix) const;

        std::unique_ptr<ILabelSamplingFactory> createLabelSamplingFactory(const ILabelMatrix& labelMatrix) const;

        std::unique_ptr<IInstanceSamplingFactory> createInstanceSamplingFactory() const;

        std::unique_ptr<IFeatureSamplingFactory> createFeatureSamplingFactory(
          const IFeatureMatrix& featureMatrix) const;

        std::unique_ptr<IPartitionSamplingFactory> createPartitionSamplingFactory() const;

        std::unique_ptr<IRulePruningFactory> createRulePruningFactory() const;

        std::unique_ptr<IPostProcessorFactory> createPostProcessorFactory() const;

        std::unique_ptr<IStoppingCriterionFactory> createSizeStoppingCriterionFactory() const;

        std::unique_ptr<IStoppingCriterionFactory> createTimeStoppingCriterionFactory() const;

        std::unique_ptr<IStoppingCriterionFactory> createGlobalPruningFactory() const;

        std::unique_ptr<IPostOptimizationPhaseFactory> createSequentialPostOptimizationFactory() const;

        std::unique_ptr<IPostOptimizationPhaseFactory> createUnusedRuleRemovalFactory() const;

        std::unique_ptr<IMarginalProbabilityCalibratorFactory> createMarginalProbabilityCalibratorFactory() const;

        std::unique_ptr<IJointProbabilityCalibratorFactory> createJointProbabilityCalibratorFactory() const;

    protected:

        /**
         * May be overridden by subclasses in order create objects of the type `IStoppingCriterionFactory` to be used by
         * the rule learner.
         *
         * @param factory A reference to an object of type `StoppingCriterionListFactory` the objects may be added to
         */
        virtual void createStoppingCriterionFactories(StoppingCriterionListFactory& factory) const;

        /**
         * May be overridden by subclasses in order to create objects of the type `IPostOptimizationPhaseFactory` to be
         * used by the rule learner.
         *
         * @param factory A reference to an object of type `PostOptimizationPhaseListFactory` the objects may be added
         *                to
         */
        virtual void createPostOptimizationPhaseFactories(PostOptimizationPhaseListFactory& factory) const;

        /**
         * Must be implemented by subclasses in order to create the `IStatisticsProviderFactory` to be used by the rule
         * learner.
         *
         * @param featureMatrix A reference to an object of type `IFeatureMatrix` that provides access to the feature
         *                      values of the training examples
         * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access to
         *                      the labels of the training examples
         * @return              An unique pointer to an object of type `IStatisticsProviderFactory` that has been
         *                      created
         */
        virtual std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
          const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const = 0;

        /**
         * Must be implemented by subclasses in order to create the `IModelBuilderFactory` to be used by the rule
         * learner.
         *
         * @return An unique pointer to an object of type `IModelBuilderFactory` that has been created
         */
        virtual std::unique_ptr<IModelBuilderFactory> createModelBuilderFactory() const = 0;

        /**
         * May be overridden by subclasses in order to create the `ILabelSpaceInfo` to be used by the rule learner as a
         * basis for for making predictions.
         *
         * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access to
         *                      the labels of the training examples
         * @return              An unique pointer to an object of type `ILabelSpaceInfo` that has been created
         */
        virtual std::unique_ptr<ILabelSpaceInfo> createLabelSpaceInfo(const IRowWiseLabelMatrix& labelMatrix) const;

        /**
         * May be overridden by subclasses in order to create the `IBinaryPredictorFactory` to be used by the rule
         * learner for predicting binary labels.
         *
         * @param featureMatrix A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `IBinaryPredictorFactory` that has been created
         *                      or a null pointer, if the rule learner does not support to predict binary labels
         */
        virtual std::unique_ptr<IBinaryPredictorFactory> createBinaryPredictorFactory(
          const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const;

        /**
         * May be overridden by subclasses in order to create the `ISparseBinaryPredictorFactory` to be used by the rule
         * learner for predicting sparse binary labels.
         *
         * @param featureMatrix A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `ISparseBinaryPredictorFactory` that has been
         *                      created or a null pointer, if the rule learner does not support to predict sparse binary
         *                      labels
         */
        virtual std::unique_ptr<ISparseBinaryPredictorFactory> createSparseBinaryPredictorFactory(
          const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const;

        /**
         * May be overridden by subclasses in order to create the `IScorePredictorFactory` to be used by the rule
         * learner for predicting regression scores.
         *
         * @param featureMatrix A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `IScorePredictorFactory` that has been created or
         *                      a null pointer, if the rule learner does not support to predict regression scores
         */
        virtual std::unique_ptr<IScorePredictorFactory> createScorePredictorFactory(
          const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const;

        /**
         * May be overridden by subclasses in order to create the `IProbabilityPredictorFactory` to be used by the rule
         * learner for predicting probability estimates.
         *
         * @param featureMatrix A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `IProbabilityPredictorFactory` that has been
         *                      created or a null pointer, if the rule learner does not support to predict probability
         *                      estimates
         */
        virtual std::unique_ptr<IProbabilityPredictorFactory> createProbabilityPredictorFactory(
          const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const;

    public:

        /**
         * @param config A reference to an object of type `IRuleLearner::IConfig` that specifies the configuration that
         *               should be used by the rule learner
         */
        AbstractRuleLearner(IRuleLearner::IConfig& config);

        std::unique_ptr<ITrainingResult> fit(const IFeatureInfo& featureInfo,
                                             const IColumnWiseFeatureMatrix& featureMatrix,
                                             const IRowWiseLabelMatrix& labelMatrix, uint32 randomState) const override;

        bool canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix,
                              const ITrainingResult& trainingResult) const override;

        bool canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                                const ITrainingResult& trainingResult) const override;

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const override;

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        bool canPredictScores(const IRowWiseFeatureMatrix& featureMatrix,
                              const ITrainingResult& trainingResult) const override;

        bool canPredictScores(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

        std::unique_ptr<IScorePredictor> createScorePredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                              const ITrainingResult& trainingResult) const override;

        std::unique_ptr<IScorePredictor> createScorePredictor(const IRowWiseFeatureMatrix& featureMatrix,
                                                              const IRuleModel& ruleModel,
                                                              const ILabelSpaceInfo& labelSpaceInfo,
                                                              uint32 numLabels) const override;

        bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix,
                                     const ITrainingResult& trainingResult) const override;

        bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const ITrainingResult& trainingResult) const override;

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;
};
