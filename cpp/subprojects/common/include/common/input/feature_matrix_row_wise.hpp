/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_matrix.hpp"

#include <memory>

// Forward declarations
class IRuleModel;
class ILabelSpaceInfo;
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
 * Defines an interface for all feature matrices that provide row-wise access to the feature values of examples.
 */
class MLRLCOMMON_API IRowWiseFeatureMatrix : virtual public IFeatureMatrix {
    public:

        virtual ~IRowWiseFeatureMatrix() override {};

        /**
         * Creates and returns a new instance of the class `IBinaryPredictor`, based on the type of this feature matrix.
         *
         * @param factory                             A reference to an object of type `IBinaryPredictorFactory` that
         *                                            should be used to create the instance
         * @param ruleModel                           A reference to an object of type `IRuleModel` that should be used
         *                                            to obtain predictions
         * @param labelSpaceInfo                      A reference to an object of type `ILabelSpaceInfo` that provides
         *                                            information about the label space that may be used as a basis for
         *                                            making predictions
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
          const IBinaryPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `ISparseBinaryPredictor`, based on the type of this feature
         * matrix.
         *
         * @param factory                             A reference to an object of type `ISparseBinaryPredictorFactory`
         *                                            that should be used to create the instance
         * @param ruleModel                           A reference to an object of type `IRuleModel` that should be used
         *                                            to obtain predictions
         * @param labelSpaceInfo                      A reference to an object of type `ILabelSpaceInfo` that provides
         *                                            information about the label space that may be used as a basis for
         *                                            making predictions
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
          const ISparseBinaryPredictorFactory& factory, const IRuleModel& ruleModel,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IScorePredictor`, based on the type of this feature matrix.
         *
         * @param factory         A reference to an object of type `IScorePredictorFactory` that should be used to
         *                        create the instance
         * @param ruleModel       A reference to an object of type `IRuleModel` that should be used to obtain
         *                        predictions
         * @param labelSpaceInfo  A reference to an object of type `ILabelSpaceInfo` that provides information about the
         *                        label space that may be used as a basis for making predictions
         * @param numLabels       The number of labels to predict for
         * @return                An unique pointer to an object of type `IScorePredictor` that has been created
         */
        virtual std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                                      const IRuleModel& ruleModel,
                                                                      const ILabelSpaceInfo& labelSpaceInfo,
                                                                      uint32 numLabels) const = 0;

        /**
         * Creates and returns a new instance of the class `IProbabilityPredictor`, based on the type of this feature
         * matrix.
         *
         * @param factory                             A reference to an object of type `IProbabilityPredictorFactory`
         *                                            that should be used to create the instance
         * @param ruleModel                           A reference to an object of type `IRuleModel` that should be used
         *                                            to obtain predictions
         * @param labelSpaceInfo                      A reference to an object of type `ILabelSpaceInfo` that provides
         *                                            information about the label space that may be used as a basis for
         *                                            making predictions
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
          const IProbabilityPredictorFactory& factory, const IRuleModel& ruleModel,
          const ILabelSpaceInfo& labelSpaceInfo,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;
};
