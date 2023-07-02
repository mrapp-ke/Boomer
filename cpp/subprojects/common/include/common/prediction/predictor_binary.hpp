/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"
#include "common/data/view_csr.hpp"
#include "common/model/rule_list.hpp"
#include "common/prediction/label_vector_set.hpp"
#include "common/prediction/prediction_matrix_dense.hpp"
#include "common/prediction/prediction_matrix_sparse_binary.hpp"
#include "common/prediction/predictor.hpp"
#include "common/prediction/probability_calibration_marginal.hpp"

/**
 * Defines an interface for all classes that allow to predict binary labels for given query examples.
 */
class IBinaryPredictor : virtual public IPredictor<DensePredictionMatrix<uint8>> {
    public:

        virtual ~IBinaryPredictor() override {};
};

/**
 * Defines an interface for all classes that allow to create instances of the type `IBinaryPredictor`.
 */
class IBinaryPredictorFactory {
    public:

        virtual ~IBinaryPredictorFactory() {};

        /**
         * Creates and returns a new object of the type `IBinaryPredictor`.
         *
         * @param featureMatrix                         A reference to an object of type `CsrConstView` that stores the
         *                                              feature values of the query examples to predict for
         * @param model                                 A reference to an object of type `RuleList` that should be used
         *                                              to obtain predictions
         * @param labelVectorSet                        A pointer to an object of type `LabelVectorSet` that stores all
         *                                              known label vectors or a null pointer, if no such set is
         *                                              available
         * @param marginalProbabilityCalibrationModel   A reference to an object of type
         *                                              `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                              calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel      A reference to an object of type
         *                                              `IJointProbabilityCalibrationModel` that may be used for the
         *                                              calibration of joint probabilities
         * @param numLabels                             The number of labels to predict for
         * @return                                      An unique pointer to an object of type `IBinaryPredictor` that
         *                                              has been created
         */
        virtual std::unique_ptr<IBinaryPredictor> create(
          const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
          const LabelVectorSet* labelVectorSet,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new object of the type `IBinaryPredictor`.
         *
         * @param featureMatrix                         A reference to an object of type `CsrConstView` that stores the
         *                                              feature values of the query examples to predict for
         * @param model                                 A reference to an object of type `RuleList` that should be used
         *                                              to obtain predictions
         * @param labelVectorSet                        A pointer to an object of type `LabelVectorSet` that stores all
         *                                              known label vectors or a null pointer, if no such set is
         *                                              available
         * @param marginalProbabilityCalibrationModel   A reference to an object of type
         *                                              `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                              calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel      A reference to an object of type
         *                                              `IJointProbabilityCalibrationModel` that may be used for the
         *                                              calibration of joint probabilities
         * @param numLabels                             The number of labels to predict for
         * @return                                      An unique pointer to an object of type `IBinaryPredictor` that
         *                                              has been created
         */
        virtual std::unique_ptr<IBinaryPredictor> create(
          const CsrConstView<const float32>& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;
};

/**
 * Defines an interface for all classes that allow to predict sparse binary labels for given query examples.
 */
class ISparseBinaryPredictor : public IPredictor<BinarySparsePredictionMatrix> {
    public:

        virtual ~ISparseBinaryPredictor() override {};
};

/**
 * Defines an interface for all classes that allow to create instances of the type `ISparseBinaryPredictor`.
 */
class ISparseBinaryPredictorFactory {
    public:

        virtual ~ISparseBinaryPredictorFactory() {};

        /**
         * Creates and returns a new object of the type `ISparseBinaryPredictor`.
         *
         * @param featureMatrix                         A reference to an object of type `CsrConstView` that stores the
         *                                              feature values of the query examples to predict for
         * @param model                                 A reference to an object of type `RuleList` that should be used
         *                                              to obtain predictions
         * @param labelVectorSet                        A pointer to an object of type `LabelVectorSet` that stores all
         *                                              known label vectors or a null pointer, if no such set is
         *                                              available
         * @param marginalProbabilityCalibrationModel   A reference to an object of type
         *                                              `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                              calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel      A reference to an object of type
         *                                              `IJointProbabilityCalibrationModel` that may be used for the
         *                                              calibration of joint probabilities
         * @param numLabels                             The number of labels to predict for
         * @return                                      An unique pointer to an object of type `ISparseBinaryPredictor`
         *                                              that has been created
         */
        virtual std::unique_ptr<ISparseBinaryPredictor> create(
          const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
          const LabelVectorSet* labelVectorSet,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new object of the type `ISparseBinaryPredictor`.
         *
         * @param featureMatrix                         A reference to an object of type `CsrConstView` that stores the
         *                                              feature values of the query examples to predict for
         * @param model                                 A reference to an object of type `RuleList` that should be used
         *                                              to obtain predictions
         * @param labelVectorSet                        A pointer to an object of type `LabelVectorSet` that stores all
         *                                              known label vectors or a null pointer, if no such set is
         *                                              available
         * @param marginalProbabilityCalibrationModel   A reference to an object of type
         *                                              `IMarginalProbabilityCalibrationModel` that may be used for the
         *                                              calibration of marginal probabilities
         * @param jointProbabilityCalibrationModel      A reference to an object of type
         *                                              `IJointProbabilityCalibrationModel` that may be used for the
         *                                              calibration of joint probabilities
         * @param numLabels                             The number of labels to predict for
         * @return                                      An unique pointer to an object of type `ISparseBinaryPredictor`
         *                                              that has been created
         */
        virtual std::unique_ptr<ISparseBinaryPredictor> create(
          const CsrConstView<const float32>& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure an `IBinaryPredictor` or `ISparseBinaryPredictor`.
 */
class IBinaryPredictorConfig : public IPredictorConfig<IBinaryPredictorFactory> {
    public:

        virtual ~IBinaryPredictorConfig() override {};

        /**
         * Creates and returns a new object of type `ISparseBinaryPredictorFactory` according to the specified
         * configuration.
         *
         * @param featureMatrix A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples to predict for
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `ISparseBinaryPredictorFactory` that has been
         *                      created
         */
        virtual std::unique_ptr<ISparseBinaryPredictorFactory> createSparsePredictorFactory(
          const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const = 0;
};
