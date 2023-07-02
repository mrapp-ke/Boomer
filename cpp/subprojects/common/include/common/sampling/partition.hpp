/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/util/quality.hpp"

#include <memory>

// Forward declarations
class IStoppingCriterion;
class IStoppingCriterionFactory;
class IInstanceSampling;
class IInstanceSamplingFactory;
class IRowWiseLabelMatrix;
class IStatistics;
class IThresholdsSubset;
class ICoverageState;
class AbstractPrediction;
class IMarginalProbabilityCalibrationModel;
class IMarginalProbabilityCalibrator;
class IJointProbabilityCalibrationModel;
class IJointProbabilityCalibrator;

/**
 * Defines an interface for all classes that provide access to the indices of training examples that have been split
 * into a training set and a holdout set.
 */
class IPartition {
    public:

        virtual ~IPartition() {};

        /**
         * Creates and returns a new instance of the class `IStoppingCriterion`, based on the type of this partition.
         *
         * @param factory   A reference to an object of type `IStoppingCriterionFactory` that should be used to create
         *                  the instance
         * @return          An unique pointer to an object of type `IStoppingCriterion` that has been created
         */
        virtual std::unique_ptr<IStoppingCriterion> createStoppingCriterion(
          const IStoppingCriterionFactory& factory) = 0;

        /**
         * Creates and returns a new instance of the class `IInstanceSampling`, based on the type of this partition.
         *
         * @param factory       A reference to an object of type `IInstanceSamplingFactory` that should be used to
         *                      create the instance
         * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access to
         *                      the labels of individual training examples
         * @param statistics    A reference to an object of type `IStatistics` that provides access to the statistics
         *                      which serve as a basis for learning rules
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                          const IRowWiseLabelMatrix& labelMatrix,
                                                                          IStatistics& statistics) = 0;

        /**
         * Calculates and returns a numerical score that assesses the quality of a rule's prediction for all examples
         * that do not belong to the current sample and are marked as covered according to a given object of type
         * `ICoverageState`.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to
         *                          evaluate the prediction
         * @param coverageState     A reference to an object of type `ICoverageState` that keeps track of the examples
         *                          that are covered by the rule
         * @param head              A reference to an object of type `AbstractPrediction` that stores the scores that
         *                          are predicted by the rule
         * @return                  An object of type `Quality` that stores the calculated quality
         */
        virtual Quality evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset,
                                            const ICoverageState& coverageState, const AbstractPrediction& head) = 0;

        /**
         * Recalculates and updates a rule's prediction based on all examples in the training set that are marked as
         * covered according to a given object of type `ICoverageState`.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset` that should be used to
         *                          recalculate the prediction
         * @param coverageState     A reference to an object of type `ICoverageState` that keeps track of the examples
         *                          that are covered by the rule
         * @param head              A reference to an object of type `AbstractPrediction` to be updated
         */
        virtual void recalculatePrediction(const IThresholdsSubset& thresholdsSubset,
                                           const ICoverageState& coverageState, AbstractPrediction& head) = 0;

        /**
         * Fits and returns a model for the calibration of marginal probabilities, based on the type of this partition.
         *
         * @param probabilityCalibrator A reference to an object of type `IMarginalProbabilityCalibrator` that should be
         *                              used to fit the calibration model
         * @param labelMatrix           A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise
         *                              access to the labels of the training examples
         * @param statistics            A reference to an object of type `IStatistics` that provides access to
         *                              statistics about the labels of the training examples
         * @return                      An unique pointer to an object of type `IMarginalProbabilityCalibrationModel`
         *                              that has been fit
         */
        virtual std::unique_ptr<IMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
          const IMarginalProbabilityCalibrator& probabilityCalibrator, const IRowWiseLabelMatrix& labelMatrix,
          const IStatistics& statistics) = 0;

        /**
         * Fits and returns a model for the calibration of joint probabilities, based on the type of this partition.
         *
         * @param probabilityCalibrator A reference to an object of type `IJointProbabilityCalibrator` that should be
         *                              used to fit the calibration model
         * @param labelMatrix           A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise
         *                              access to the labels of the training examples
         * @param statistics            A reference to an object of type `IStatistics` that provides access to
         *                              statistics about the labels of the training examples
         * @return                      An unique pointer to an object of type `IJointProbabilityCalibrationModel` that
         *                              has been fit
         */
        virtual std::unique_ptr<IJointProbabilityCalibrationModel> fitJointProbabilityCalibrationModel(
          const IJointProbabilityCalibrator& probabilityCalibrator, const IRowWiseLabelMatrix& labelMatrix,
          const IStatistics& statistics) = 0;
};
