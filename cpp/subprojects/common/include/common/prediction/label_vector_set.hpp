/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/arrays.hpp"
#include "common/input/label_matrix_row_wise.hpp"
#include "common/prediction/label_space_info.hpp"

#include <functional>
#include <vector>

/**
 * Defines an interface for all classes that provide access to a set of unique label vectors.
 */
class MLRLCOMMON_API ILabelVectorSet : public ILabelSpaceInfo {
    public:

        virtual ~ILabelVectorSet() override {};

        /**
         * A visitor function for handling objects of the type `LabelVector` and their frequencies.
         */
        typedef std::function<void(const LabelVector&, uint32)> LabelVectorVisitor;

        /**
         * Adds a label vector to the set.
         *
         * @param labelVectorPtr    An unique pointer to an object of type `LabelVector`
         * @param frequency         The frequency of the label vector
         */
        virtual void addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr, uint32 frequency) = 0;

        /**
         * Invokes the given visitor function for each label vector that has been added to the set.
         *
         * @param visitor The visitor function for handling objects of the type `LabelVector`
         */
        virtual void visit(LabelVectorVisitor visitor) const = 0;
};

/**
 * An implementation of the type `ILabelVectorSet` that stores a set of unique label vectors, as well as their
 * frequency.
 */
class LabelVectorSet final : public ILabelVectorSet {
    private:

        std::vector<std::unique_ptr<LabelVector>> labelVectors_;

        std::vector<uint32> frequencies_;

    public:

        LabelVectorSet();

        /**
         * @param labelMatrix A reference to an object of type `IRowWiseLabelMatrix` that stores the label vectors that
         *                    should be added to the set
         */
        LabelVectorSet(const IRowWiseLabelMatrix& labelMatrix);

        /**
         * An iterator that provides read-only access to the label vectors.
         */
        typedef std::vector<std::unique_ptr<LabelVector>>::const_iterator const_iterator;

        /**
         * An iterator that provides read-only access to the frequency of the label lectors.
         *
         */
        typedef std::vector<uint32>::const_iterator frequency_const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the label vectors in the set.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the label vectors in the set.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Returns a `frequency_const_iterator` to the beginning of the frequencies.
         *
         * @return frequency_const_iterator A `frequency_const_iterator` to the beginning
         */
        frequency_const_iterator frequencies_cbegin() const;

        /**
         * Returns a `frequency_const_iterator` to the end of the frequencies.
         *
         * @return frequency_const_iterator A `frequency_const_iterator` to the end
         */
        frequency_const_iterator frequencies_cend() const;

        /**
         * Returns the number of label vectors in the set.
         *
         * @return The number of label vectors
         */
        uint32 getNumLabelVectors() const;

        void addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr, uint32 frequency) override;

        void visit(LabelVectorVisitor visitor) const override;

        std::unique_ptr<IJointProbabilityCalibrator> createJointProbabilityCalibrator(
          const IJointProbabilityCalibratorFactory& factory,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const override;

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
          const RuleList& ruleList, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<IBinaryPredictor> createBinaryPredictor(
          const IBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& ruleList,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
          const RuleList& ruleList, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<ISparseBinaryPredictor> createSparseBinaryPredictor(
          const ISparseBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& ruleList,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CContiguousFeatureMatrix& featureMatrix,
                                                              const RuleList& ruleList,
                                                              uint32 numLabels) const override;

        std::unique_ptr<IScorePredictor> createScorePredictor(const IScorePredictorFactory& factory,
                                                              const CsrFeatureMatrix& featureMatrix,
                                                              const RuleList& ruleList,
                                                              uint32 numLabels) const override;

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
          const RuleList& ruleList, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
          const IProbabilityPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& ruleList,
          const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
          const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const override;
};

/**
 * Creates and returns a new object of the type `ILabelVectorSet`.
 *
 * @return An unique pointer to an object of type `ILabelVectorSet` that has been created
 */
MLRLCOMMON_API std::unique_ptr<ILabelVectorSet> createLabelVectorSet();
