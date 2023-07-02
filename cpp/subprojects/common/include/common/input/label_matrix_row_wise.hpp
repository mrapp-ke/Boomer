/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/label_matrix.hpp"
#include "common/input/label_vector.hpp"

#include <memory>

// Forward declarations
class IStatisticsProvider;
class IStatisticsProviderFactory;
class IPartitionSampling;
class IPartitionSamplingFactory;
class IInstanceSampling;
class IInstanceSamplingFactory;
class IStatistics;
class SinglePartition;
class BiPartition;
class IMarginalProbabilityCalibrator;
class IMarginalProbabilityCalibrationModel;
class IJointProbabilityCalibrator;
class IJointProbabilityCalibrationModel;

/**
 * Defines an interface for all label matrices that provide access to the labels of the training examples.
 */
class MLRLCOMMON_API IRowWiseLabelMatrix : virtual public ILabelMatrix {
    public:

        virtual ~IRowWiseLabelMatrix() override {};

        /**
         * Calculates and returns the label cardinality, i.e., the average number of relevant labels per example.
         *
         * @return The label cardinality
         */
        virtual float32 calculateLabelCardinality() const = 0;

        /**
         * Creates and returns a label vector that corresponds to a specific row in the label matrix.
         *
         * @param row   The row
         * @return      An unique pointer to an object of type `LabelVector` that has been created
         */
        virtual std::unique_ptr<LabelVector> createLabelVector(uint32 row) const = 0;

        /**
         * Creates and returns a new instance of the class `IStatisticsProvider`, based on the type of this label
         * matrix.
         *
         * @param factory   A reference to an object of type `IStatisticsProviderFactory` that should be used to create
         *                  the instance
         * @return          An unique pointer to an object of type `IStatisticsProvider` that has been created
         */
        virtual std::unique_ptr<IStatisticsProvider> createStatisticsProvider(
          const IStatisticsProviderFactory& factory) const = 0;

        /**
         * Creates and returns a new instance of the class `IPartitionSampling`, based on the type of this label matrix.
         *
         * @param factory   A reference to an object of type `IPartitionSamplingFactory` that should be used to create
         *                  the instance
         * @return          An unique pointer to an object of type `IPartitionSampling` that has been created
         */
        virtual std::unique_ptr<IPartitionSampling> createPartitionSampling(
          const IPartitionSamplingFactory& factory) const = 0;

        /**
         * Creates and returns a new instance of the class `IInstanceSampling`, based on the type of this label matrix.
         *
         * @param factory       A reference to an object of type `IInstanceSamplingFactory` that should be used to
         *                      create the instance
         * @param partition     A reference to an object of type `SinglePartition` that provides access to the indices
         *                      of the training examples that are included in the training set
         * @param statistics    A reference to an object of type `IStatistics` that provides access to the statistics
         *                      which serve as a basis for learning rules
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                          const SinglePartition& partition,
                                                                          IStatistics& statistics) const = 0;

        /**
         * Creates and returns a new instance of the class `IInstanceSampling`, based on the type of this label matrix.
         *
         * @param factory       A reference to an object of type `IInstanceSamplingFactory` that should be used to
         *                      create the instance
         * @param partition     A reference to an object of type `BiPartition` that provides access to the indices of
         *                      the training examples that are included in the training set and the holdout set,
         *                      respectively
         * @param statistics    A reference to an object of type `IStatistics` that provides access to the statistics
         *                      which serve as a basis for learning rules
         * @return              An unique pointer to an object of type `IInstanceSampling` that has been created
         */
        virtual std::unique_ptr<IInstanceSampling> createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                          BiPartition& partition,
                                                                          IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of marginal probabilities, based on the type of this label
         * matrix.
         *
         * @param probabilityCalibrator A reference to an object of type `IMarginalProbabilityCalibrator` that should be
         *                              used to fit the calibration model
         * @param partition             A reference to an object of type `SinglePartition` that provides access to the
         *                              indices of the training examples that are included in the training set
         * @param statistics            A reference to an object of type `IStatistics` that provides access to
         *                              statistics about the labels of the training examples
         * @return                      An unique pointer to an object of type `IMarginalProbabilityCalibrationModel`
         *                              that has been fit
         */
        virtual std::unique_ptr<IMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
          const IMarginalProbabilityCalibrator& probabilityCalibrator, const SinglePartition& partition,
          const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of marginal probabilities, based on the type of this label
         * matrix.
         *
         * @param probabilityCalibrator A reference to an object of type `IMarginalProbabilityCalibrator` that should be
         *                              used to fit the calibration model
         * @param partition             A reference to an object of type `BiPartition` that provides access to the
         *                              indices of the training examples that are included in the training set and the
         *                              holdout set, respectively
         * @param statistics            A reference to an object of type `IStatistics` that provides access to
         *                              statistics about the labels of the training examples
         * @return                      An unique pointer to an object of type `IMarginalProbabilityCalibrationModel`
         *                              that has been fit
         */
        virtual std::unique_ptr<IMarginalProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
          const IMarginalProbabilityCalibrator& probabilityCalibrator, BiPartition& partition,
          const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of joint probabilities, based on the type of this label matrix.
         *
         * @param probabilityCalibrator A reference to an object of type `IJointProbabilityCalibrator` that should be
         *                              used to fit the calibration model
         * @param partition             A reference to an object of type `SinglePartition` that provides access to the
         *                              indices of the training examples that are included in the training set
         * @param statistics            A reference to an object of type `IStatistics` that provides access to
         *                              statistics about the labels of the training examples
         * @return                      An unique pointer to an object of type `IJointProbabilityCalibrationModel` that
         *                              has been fit
         */
        virtual std::unique_ptr<IJointProbabilityCalibrationModel> fitJointProbabilityCalibrationModel(
          const IJointProbabilityCalibrator& probabilityCalibrator, const SinglePartition& partition,
          const IStatistics& statistics) const = 0;

        /**
         * Fits and returns a model for the calibration of joint probabilities, based on the type of this label matrix.
         *
         * @param probabilityCalibrator A reference to an object of type `IJointProbabilityCalibrator` that should be
         *                              used to fit the calibration model
         * @param partition             A reference to an object of type `BiPartition` that provides access to the
         *                              indices of the training examples that are included in the training set and the
         *                              holdout set, respectively
         * @param statistics            A reference to an object of type `IStatistics` that provides access to
         *                              statistics about the labels of the training examples
         * @return                      An unique pointer to an object of type `IJointProbabilityCalibrationModel` that
         *                              has been fit
         */
        virtual std::unique_ptr<IJointProbabilityCalibrationModel> fitJointProbabilityCalibrationModel(
          const IJointProbabilityCalibrator& probabilityCalibrator, BiPartition& partition,
          const IStatistics& statistics) const = 0;
};
