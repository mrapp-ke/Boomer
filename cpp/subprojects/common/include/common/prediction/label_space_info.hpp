/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include "common/macros.hpp"

#include <memory>

class IJointProbabilityCalibrator;
class IJointProbabilityCalibratorFactory;
class CContiguousFeatureMatrix;
class CsrFeatureMatrix;
class RuleList;
class IMarginalProbabilityCalibrationModel;
class IJointProbabilityCalibrationModel;
class IBinaryPredictor;
class IBinaryPredictorFactory;
class ISparseBinaryPredictor;
class ISparseBinaryPredictorFactory;
class IScorePredictor;
class IScorePredictorFactory;
class IProbabilityPredictor;
class IProbabilityPredictorFactory;

/**
 * Defines an interface for all classes that provide information about the label space that may be used as a basis for
 * making predictions.
 */
class MLRLCOMMON_API ILabelSpaceInfo {
    public:

        virtual ~ILabelSpaceInfo() {};

        /**
         * Creates and returns a new instance of the class `IJointProbabilityCalibrator`, based on the type of this
         * information about the label space.
         *
         * @param factory                             A reference to an object of type
         *                                            `IJointProbabilityCalibratorFactory` that should be used to create
         *                                            the instance
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @return                                    An unique pointer to an object of type
         *                                            `IJointProbabilityPredictor` that has been created
         */
        virtual std::unique_ptr<IJointProbabilityCalibrator> createJointProbabilityCalibrator(
          const IJointProbabilityCalibratorFactory& factory,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const = 0;

        /**
         * Creates and returns a new instance of the class `IBinaryPredictor`, based on the type of this information
         * about the label space.
         *
         * @param factory                             A reference to an object of type `IBinaryPredictorFactory` that
         *                                            should be used to create the instance
         * @param featureMatrix                       A reference to an object of type `CContiguousFeatureMatrix` that
         *                                            provides row-wise access to the features of the query examples
         * @param model                               A reference to an object of type `RuleList` that should be used to
         *                                            obtain predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `IBinaryPredictor` that has
         *                                            been created
         */
        virtual std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix, const RuleList& model,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IBinaryPredictor`, based on the type of this information
         * about the label space.
         *
         * @param factory                             A reference to an object of type `IBinaryPredictorFactory` that
         *                                            should be used to create the instance
         * @param featureMatrix                       A reference to an object of type `CsrFeatureMatrix` that provides
         *                                            row-wise access to the features of the query examples
         * @param model                               A reference to an object of type `RuleList` that should be used to
         *                                            obtain predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `IBinaryPredictor` that has
         *                                            been created
         */
        virtual std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `ISparseBinaryPredictor`, based on the type of this
         * information about the label space.
         *
         * @param factory                             A reference to an object of type `ISparseBinaryPredictorFactory`
         *                                            that should be used to create the instance
         * @param featureMatrix                       A reference to an object of type `CContiguousFeatureMatrix` that
         *                                            provides row-wise access to the features of the query examples
         * @param model                               A reference to an object of type `RuleList` that should be used to
         *                                            obtain predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `ISparseBinaryPredictor`
         *                                            that has been created
         */
        virtual std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `ISparseBinaryPredictor`, based on the type of this
         * information about the label space.
         *
         * @param factory                             A reference to an object of type `ISparseBinaryPredictorFactory`
         *                                            that should be used to create the instance
         * @param featureMatrix                       A reference to an object of type `CsrFeatureMatrix` that provides
         *                                            row-wise access to the features of the query examples
         * @param model                               A reference to an object of type `RuleList` that should be used to
         *                                            obtain predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `ISparseBinaryPredictor`
         *                                            that has been created
         */
        virtual std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IScorePredictor`, based on the type of this information
         * about the label space.
         *
         * @param factory       A reference to an object of type `IScorePredictorFactory` that should be used to create
         *                      the instance
         * @param featureMatrix A reference to an object of type `CContiguousFeatureMatrix` that provides row-wise
         *                      access to the features of the query examples
         * @param model         A reference to an object of type `RuleList` that should be used to obtain predictions
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `IScorePredictor` that has been created
         */
        virtual std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                                      const CContiguousFeatureMatrix& featureMatrix,
                                                                      const RuleList& model,
                                                                      uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IScorePredictor`, based on the type of this information
         * about the label space.
         *
         * @param factory       A reference to an object of type `IScorePredictorFactory` that should be used to create
         *                      the instance
         * @param featureMatrix A reference to an object of type `CsrFeatureMatrix` that provides row-wise access to the
         *                      features of the query examples
         * @param model         A reference to an object of type `RuleList` that should be used to obtain predictions
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `IScorePredictor` that has been created
         */
        virtual std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                                      const CsrFeatureMatrix& featureMatrix,
                                                                      const RuleList& model,
                                                                      uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IProbabilityPredictor`, based on the type of this
         * information about the label space.
         *
         * @param factory                             A reference to an object of type `IProbabilityPredictorFactory`
         *                                            that should be used to create the instance
         * @param featureMatrix                       A reference to an object of type `CContiguousFeatureMatrix` that
         *                                            provides row-wise access to the features of the query examples
         * @param model                               A reference to an object of type `RuleList` that should be used to
         *                                            obtain predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `IProbabilityPredictor`
         *                                            that has been created
         */
        virtual std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
          const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IProbabilityPredictor`, based on the type of this
         * information about the label space.
         *
         * @param factory                             A reference to an object of type `IProbabilityPredictorFactory`
         *                                            that should be used to create the instance
         * @param featureMatrix                       A reference to an object of type `CsrFeatureMatrix` that provides
         *                                            row-wise access to the features of the query examples
         * @param model                               A reference to an object of type `RuleList` that should be used to
         *                                            obtain predictions
         * @param marginalProbabilityCalibrationModel A reference to an object of type
         *                                            `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                            calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel    A reference to an object of type
         *                                            `IJointProbabilityCalibrationModel` that may be used for the
         *                                            calibration of joint probabilities
         * @param numLabels                           The number of labels to predict for
         * @return                                    An unique pointer to an object of type `IProbabilityPredictor`
         *                                            that has been created
         */
        virtual std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;
};
