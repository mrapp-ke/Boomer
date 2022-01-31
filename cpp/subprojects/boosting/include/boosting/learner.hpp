/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#ifdef _WIN32
    #pragma warning( push )
    #pragma warning( disable : 4250 )
#endif

#include "common/learner.hpp"
#include "boosting/binning/label_binning_equal_width.hpp"
#include "boosting/losses/loss.hpp"
#include "boosting/math/blas.hpp"
#include "boosting/math/lapack.hpp"
#include "boosting/post_processing/shrinkage_constant.hpp"
#include "boosting/rule_evaluation/head_type.hpp"
#include "boosting/rule_evaluation/regularization_manual.hpp"


namespace boosting {

    /**
     * Defines an interface for all rule learners that make use of gradient boosting.
     */
    class MLRLBOOSTING_API IBoostingRuleLearner : virtual public IRuleLearner {

        public:

            /**
             * Defines an interface for all classes that allow to configure a rule learner that makes use of gradient
             * boosting.
             */
            class IConfig : virtual public IRuleLearner::IConfig {

                friend class BoostingRuleLearner;

                private:

                    /**
                     * Returns the configuration of the rule heads that should be induced by the rule learner.
                     *
                     * @return A reference to an object of type `IHeadConfig` that specifies the configuration of the
                     *         rule heads
                     */
                    virtual const IHeadConfig& getHeadConfig() const = 0;

                    /**
                     * Returns the configuration of the L1 regularization term.
                     *
                     * @return A reference to an object of type `IRegularizationConfig` that specifies the configuration
                     *         of the L1 regularization term
                     */
                    virtual const IRegularizationConfig& getL1RegularizationConfig() const = 0;

                    /**
                     * Returns the configuration of the L2 regularization term.
                     *
                     * @return A reference to an object of type `IRegularizationConfig` that specifies the configuration
                     *         of the L2 regularization term
                     */
                    virtual const IRegularizationConfig& getL2RegularizationConfig() const = 0;

                    /**
                     * Returns the configuration of the loss function.
                     *
                     * @return A reference to an object of type `ILossConfig` that specifies the configuration of the
                     *         loss function
                     */
                    virtual const ILossConfig& getLossConfig() const = 0;

                    /**
                     * Returns the configuration of the method for the assignment of labels to bins.
                     *
                     * @return A reference to an object of type `ILabelBinningConfig` that specifies the configuration
                     *         of the method for the assignment of labels to bins
                     */
                    virtual const ILabelBinningConfig& getLabelBinningConfig() const = 0;

                    /**
                     * Returns the configuration of the predictor that predicts whether individual labels of given query
                     * examples are relevant or irrelevant.
                     *
                     * @return A reference to an object of type `IClassificationPredictorConfig` that specifies the
                     *         configuration of the predictor that predicts whether individual labels of given query
                     *         examples are relevant or irrelevant
                     */
                    virtual const IClassificationPredictorConfig& getClassificationPredictorConfig() const = 0;

                    /**
                     * Returns the configuration of the predictor that predicts regression scores for individual labels.
                     *
                     * @return A reference to an object of type `IClassificationPredictorConfig` that specifies the
                     *         configuration of the predictor that predicts regression scores for individual labels
                     */
                    virtual const IRegressionPredictorConfig& getRegressionPredictorConfig() const = 0;

                    /**
                     * Returns the configuration of the predictor that predicts probability estimates for individual
                     * labels.
                     *
                     * @return A reference to an object of type `IClassificationPredictorConfig` that specifies the
                     *         configuration of the predictor that predicts probability estimates for individual labels
                     */
                    virtual const IProbabilityPredictorConfig& getProbabilityPredictorConfig() const = 0;

                public:

                    virtual ~IConfig() override { };

                    /**
                     * Configures the rule learning to automatically decide whether a method for the assignment of
                     * numerical feature values to bins should be used or not.
                     */
                    virtual void useAutomaticFeatureBinning() = 0;

                    /**
                     * Configures the rule learner to use a post processor that shrinks the weights of rules by a
                     * constant "shrinkage" parameter.
                     *
                     * @return A reference to an object of type `IConstantShrinkageConfig` that allows further
                     *         configuration of the loss function
                     */
                    virtual IConstantShrinkageConfig& useConstantShrinkagePostProcessor() = 0;

                    /**
                     * Configures the rule learner to automatically decide whether multi-threading should be used for
                     * the parallel refinement of rules or not.
                     */
                    virtual void useAutomaticParallelRuleRefinement() = 0;

                    /**
                     * Configures the rule learner to automatically decide whether multi-threading should be used for
                     * the parallel update of statistics or not.
                     */
                    virtual void useAutomaticParallelStatisticUpdate() = 0;

                    /**
                     * Configures the rule learner to automatically decide for the type of rule heads that should be
                     * used.
                     */
                    virtual void useAutomaticHeads() = 0;

                    /**
                     * Configures the rule learner to induce rules with single-label heads that predict for a single
                     * label.
                     */
                    virtual void useSingleLabelHeads() = 0;

                    /**
                     * Configures the rule learner to induce rules with complete heads that predict for all available
                     * labels.
                     */
                    virtual void useCompleteHeads() = 0;

                    /**
                     * Configures the rule learner to not use L1 regularization.
                     */
                    virtual void useNoL1Regularization() = 0;

                    /**
                     * Configures the rule learner to use L1 regularization.
                     *
                     * @return A reference to an object of type `IManualRegularizationConfig` that allows further
                     *         configuration of the regularization term
                     */
                    virtual IManualRegularizationConfig& useL1Regularization() = 0;

                    /**
                     * Configures the rule learner to not use L2 regularization.
                     */
                    virtual void useNoL2Regularization() = 0;

                    /**
                     * Configures the rule learner to use L2 regularization.
                     *
                     * @return A reference to an object of type `IManualRegularizationConfig` that allows further
                     *         configuration of the regularization term
                     */
                    virtual IManualRegularizationConfig& useL2Regularization() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * logistic loss that is applied example-wise.
                     */
                    virtual void useExampleWiseLogisticLoss() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * logistic loss that is applied label-wise.
                     */
                    virtual void useLabelWiseLogisticLoss() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * squared error loss that is applied label-wise.
                     */
                    virtual void useLabelWiseSquaredErrorLoss() = 0;

                    /**
                     * Configures the rule learner to use a loss function that implements a multi-label variant of the
                     * squared hinge loss that is applied label-wise.
                     */
                    virtual void useLabelWiseSquaredHingeLoss() = 0;

                    /**
                     * Configures the rule learner to not use any method for the assignment of labels to bins.
                     */
                    virtual void useNoLabelBinning() = 0;

                    /**
                     * Configures the rule learner to automatically decide whether a method for the assignment of labels
                     * to bins should be used or not.
                     */
                    virtual void useAutomaticLabelBinning() = 0;

                    /**
                     * Configures the rule learner to use a method for the assignment of labels to bins in a way such
                     * that each bin contains labels for which the predicted score is expected to belong to the same
                     * value range.
                     *
                     * @return A reference to an object of type `IEqualWidthLabelBinningConfig` that allows further
                     *         configuration of the method for the assignment of labels to bins
                     */
                    virtual IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning() = 0;

                    /**
                     * Configures the rule learner to use a predictor for predicting whether individual labels are
                     * relevant or irrelevant by summing up the scores that are provided by an existing rule-based model
                     * and comparing the aggregated score vector to the known label vectors according to a certain
                     * distance measure. The label vector that is closest to the aggregated score vector is finally
                     * predicted.
                     */
                    virtual void useExampleWiseClassificationPredictor() = 0;

                    /**
                     * Configures the rule learner to use a predictor for predicting whether individual labels are
                     * relevant or irrelevant by summing up the scores that are provided by the individual rules of an
                     * existing rule-based model and transforming them into binary values according to a certain
                     * threshold that is applied to each label individually.
                     */
                    virtual void useLabelWiseClassificationPredictor() = 0;

                    /**
                     * Configures the rule learner to automatically decide for a predictor for predicting whether
                     * individual labels are relevant or irrelevant.
                     */
                    virtual void useAutomaticClassificationPredictor() = 0;

                    /**
                     * Configures the rule learner to use a predictor for predicting regression scores by summing up the
                     * scores that are provided by the individual rules of an existing rule-based model for each label
                     * individually.
                     */
                    virtual void useLabelWiseRegressionPredictor() = 0;

                    /**
                     * Configures the rule learner to use a predictor for predicting probability estimates by summing up
                     * the scores that are provided by individual rules of an existing rule-based models and
                     * transforming the aggregated scores into probabilities according to a certain transformation
                     * function that is applied to each label individually.
                     */
                    virtual void useLabelWiseProbabilityPredictor() = 0;

            };

            virtual ~IBoostingRuleLearner() override { };

    };

    /**
     * A rule learner that makes use of gradient boosting.
     */
    class BoostingRuleLearner final : public AbstractRuleLearner, virtual public IBoostingRuleLearner {

        public:

            /**
             * Allows to configure a rule learner that makes use of gradient boosting.
             */
            class Config final : public AbstractRuleLearner::Config, virtual public IBoostingRuleLearner::IConfig {

                private:

                    std::unique_ptr<IHeadConfig> headConfigPtr_;

                    std::unique_ptr<ILossConfig> lossConfigPtr_;

                    std::unique_ptr<IRegularizationConfig> l1RegularizationConfigPtr_;

                    std::unique_ptr<IRegularizationConfig> l2RegularizationConfigPtr_;

                    std::unique_ptr<ILabelBinningConfig> labelBinningConfigPtr_;

                    std::unique_ptr<IClassificationPredictorConfig> classificationPredictorConfigPtr_;

                    std::unique_ptr<IRegressionPredictorConfig> regressionPredictorConfigPtr_;

                    std::unique_ptr<IProbabilityPredictorConfig> probabilityPredictorConfigPtr_;

                    const IHeadConfig& getHeadConfig() const override;

                    const IRegularizationConfig& getL1RegularizationConfig() const override;

                    const IRegularizationConfig& getL2RegularizationConfig() const override;

                    const ILossConfig& getLossConfig() const override;

                    const ILabelBinningConfig& getLabelBinningConfig() const override;

                    const IClassificationPredictorConfig& getClassificationPredictorConfig() const override;

                    const IRegressionPredictorConfig& getRegressionPredictorConfig() const override;

                    const IProbabilityPredictorConfig& getProbabilityPredictorConfig() const override;

                public:

                    Config();

                    /**
                     * @see `IRuleLearner::IConfig::useSizeStoppingCriterion`
                     */
                    ISizeStoppingCriterionConfig& useSizeStoppingCriterion() override;

                    void useAutomaticFeatureBinning() override final;

                    IConstantShrinkageConfig& useConstantShrinkagePostProcessor() override;

                    void useAutomaticParallelRuleRefinement() override;

                    void useAutomaticParallelStatisticUpdate() override;

                    void useAutomaticHeads() override;

                    void useSingleLabelHeads() override;

                    void useCompleteHeads() override;

                    void useNoL1Regularization() override;

                    IManualRegularizationConfig& useL1Regularization() override;

                    void useNoL2Regularization() override;

                    IManualRegularizationConfig& useL2Regularization() override;

                    void useExampleWiseLogisticLoss() override;

                    void useLabelWiseLogisticLoss() override;

                    void useLabelWiseSquaredErrorLoss() override;

                    void useLabelWiseSquaredHingeLoss() override;

                    void useNoLabelBinning() override;

                    void useAutomaticLabelBinning() override;

                    IEqualWidthLabelBinningConfig& useEqualWidthLabelBinning() override;

                    void useExampleWiseClassificationPredictor() override;

                    void useLabelWiseClassificationPredictor() override;

                    void useAutomaticClassificationPredictor() override;

                    void useLabelWiseRegressionPredictor() override;

                    void useLabelWiseProbabilityPredictor() override;

            };

        private:

            std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr_;

            Blas blas_;

            Lapack lapack_;

        protected:

            /**
             * @see `AbstractRuleLearner::createStatisticsProviderFactory`
             */
            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix) const override;

            /**
             * @see `AbstractRuleLearner::createModelBuilder`
             */
            std::unique_ptr<IModelBuilder> createModelBuilder() const override;

            /**
             * @see `AbstractRuleLearner::createLabelSpaceInfo`
             */
            std::unique_ptr<ILabelSpaceInfo> createLabelSpaceInfo(
                const IRowWiseLabelMatrix& labelMatrix) const override;

            /**
             * @see `AbstractRuleLearner::createClassificationPredictorFactory`
             */
            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory(
                const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `AbstractRuleLearner::createRegressionPredictorFactory`
             */
            std::unique_ptr<IRegressionPredictorFactory> createRegressionPredictorFactory(
                const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `AbstractRuleLearner::createProbabilityPredictorFactory`
             */
            std::unique_ptr<IProbabilityPredictorFactory> createProbabilityPredictorFactory(
                const IFeatureMatrix& featureMatrix, uint32 numLabels) const override;

        public:

            /**
             * @param configPtr     An unique pointer to an object of type `IBoostingRuleLearner::IConfig` that
             *                      specifies the configuration that should be used by the rule learner
             * @param ddotFunction  A function pointer to BLAS' DDOT routine
             * @param dspmvFunction A function pointer to BLAS' DSPMV routine
             * @param dsysvFunction A function pointer to LAPACK'S DSYSV routine
             */
            BoostingRuleLearner(std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr,
                                Blas::DdotFunction ddotFunction, Blas::DspmvFunction dspmvFunction,
                                Lapack::DsysvFunction dsysvFunction);

    };

    /**
     * Creates and returns a new object of type `IBoostingRuleLearner::IConfig`.
     *
     * @return An unique pointer to an object of type `IBoostingRuleLearner::IConfig` that has been created
     */
    MLRLBOOSTING_API std::unique_ptr<IBoostingRuleLearner::IConfig> createBoostingRuleLearnerConfig();

    /**
     * Creates and returns a new object of type `IBoostingRuleLearner`.
     *
     * @param configPtr     An unique pointer to an object of type `IBoostingRuleLearner::IConfig` that specifies the
     *                      configuration that should be used by the rule learner.
     * @param ddotFunction  A function pointer to BLAS' DDOT routine
     * @param dspmvFunction A function pointer to BLAS' DSPMV routine
     * @param dsysvFunction A function pointer to LAPACK'S DSYSV routine
     * @return              An unique pointer to an object of type `IBoostingRuleLearner` that has been created
     */
    MLRLBOOSTING_API std::unique_ptr<IBoostingRuleLearner> createBoostingRuleLearner(
        std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr, Blas::DdotFunction ddotFunction,
        Blas::DspmvFunction dspmvFunction, Lapack::DsysvFunction dsysvFunction);

}

#ifdef _WIN32
    #pragma warning ( pop )
#endif
