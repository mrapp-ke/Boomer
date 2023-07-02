/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/macros.hpp"
#include "common/post_optimization/post_optimization.hpp"

/**
 * Defines an interface for all classes that allow to configure a method that optimizes each rule in a model by
 * relearning it in the context of the other rules. Multiple iterations, where the rules in a model are relearned in the
 * order of their induction, may be carried out.
 */
class MLRLCOMMON_API ISequentialPostOptimizationConfig {
    public:

        virtual ~ISequentialPostOptimizationConfig() {};

        /**
         * Returns the number of iterations that are performed for optimizing a model.
         *
         * @return The number of iterations that are performed for optimizing a model
         */
        virtual uint32 getNumIterations() const = 0;

        /**
         * Sets the number of iterations that should be performed for optimizing a model.
         *
         * @param numIterations The number of iterations to be performed. Must be at least 1
         * @return              A reference to an object of type `ISequentialPostOptimizationConfig` that allows further
         *                      configuration of the optimization method
         */
        virtual ISequentialPostOptimizationConfig& setNumIterations(uint32 numIterations) = 0;

        /**
         * Returns whether the heads of rules are refined when being relearned or not.
         *
         * @return True, if the heads of rules are refined when being relearned, false otherwise
         */
        virtual bool areHeadsRefined() const = 0;

        /**
         * Sets whether the heads of rules should be refined when being relearned or not.
         *
         * @param refineHeads   True, if the heads of rules should be refined when being relearned, false otherwise
         * @return              A reference to an object of type `ISequentialPostOptimizationConfig` that allows further
         *                      configuration of the optimization method
         */
        virtual ISequentialPostOptimizationConfig& setRefineHeads(bool refineHeads) = 0;

        /**
         * Returns whether a new sample of the available features is created whenever a new rule is refined or not.
         *
         * @return True, if a new sample of the available features is created whenever a new rule is refined, false, if
         *         the conditions of the new rule use the same features as the original rule
         */
        virtual bool areFeaturesResampled() const = 0;

        /**
         * Sets whether a new sample of the available features should be created whenever a new rule is refined or not.
         *
         * @param resampleFeatures  True, if a new sample of the available features should be created whenever a new
         *                          rule is refined, false, if the conditions of the new rule should use the same
         *                          features as the original rule
         * @return                  A reference to an object of type `ISequentialPostOptimizationConfig` that allows
         *                          further configuration of the optimization method
         */
        virtual ISequentialPostOptimizationConfig& setResampleFeatures(bool resampleFeatures) = 0;
};

/**
 * Allows to configure a method that optimizes each rule in a model by relearning it in the context of the other rules.
 */
class SequentialPostOptimizationConfig final : public ISequentialPostOptimizationConfig,
                                               public IPostOptimizationPhaseConfig {
    private:

        uint32 numIterations_;

        bool refineHeads_;

        bool resampleFeatures_;

    public:

        SequentialPostOptimizationConfig();

        uint32 getNumIterations() const override;

        ISequentialPostOptimizationConfig& setNumIterations(uint32 numIterations) override;

        bool areHeadsRefined() const override;

        ISequentialPostOptimizationConfig& setRefineHeads(bool refineHeads) override;

        bool areFeaturesResampled() const override;

        ISequentialPostOptimizationConfig& setResampleFeatures(bool resampleFeatures) override;

        std::unique_ptr<IPostOptimizationPhaseFactory> createPostOptimizationPhaseFactory() const override;
};
