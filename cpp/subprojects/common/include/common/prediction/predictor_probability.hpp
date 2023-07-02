/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"
#include "common/data/view_csr.hpp"
#include "common/model/rule_list.hpp"
#include "common/prediction/label_vector_set.hpp"
#include "common/prediction/prediction_matrix_dense.hpp"
#include "common/prediction/predictor.hpp"
#include "common/prediction/probability_calibration_marginal.hpp"

/**
 * Defines an interface for all classes that allow to predict probability estimates for given query examples.
 */
class IProbabilityPredictor : public IPredictor<DensePredictionMatrix<float64>> {
    public:

        virtual ~IProbabilityPredictor() override {};
};

/**
 * Defines an interface for all classes that allow to create instances of the type `IProbabilityPredictor`.
 */
class IProbabilityPredictorFactory {
    public:

        virtual ~IProbabilityPredictorFactory() {};

        /**
         * Creates and returns a new object of the type `IProbabilityPredictor`.
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
         * @return                                      An unique pointer to an object of type `IProbabilityPredictor`
         *                                              that has been created
         */
        virtual std::unique_ptr<IProbabilityPredictor> create(
          const CContiguousConstView<const float32>& featureMatrix, const RuleList& model,
          const LabelVectorSet* labelVectorSet,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;

        /**
         * Creates and returns a new object of the type `IProbabilityPredictor`.
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
         * @return                                      An unique pointer to an object of type `IProbabilityPredictor`
         *                                              that has been created
         */
        virtual std::unique_ptr<IProbabilityPredictor> create(
          const CsrConstView<const float32>& featureMatrix, const RuleList& model, const LabelVectorSet* labelVectorSet,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a `IProbabilityPredictor`.
 */
class IProbabilityPredictorConfig : public IPredictorConfig<IProbabilityPredictorFactory> {
    public:

        virtual ~IProbabilityPredictorConfig() override {};
};
