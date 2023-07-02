#include "boosting/prediction/probability_calibration_isotonic.hpp"

#include "boosting/statistics/statistics.hpp"
#include "common/data/arrays.hpp"
#include "common/iterator/binary_forward_iterator.hpp"
#include "common/iterator/non_zero_index_forward_iterator.hpp"
#include "common/prediction/probability_calibration_no.hpp"

#include <algorithm>
#include <stdexcept>

namespace boosting {

    template<typename IndexIterator>
    static inline void extractThresholdsAndProbabilities(
      IndexIterator indexIterator, uint32 numExamples, uint32 numLabels,
      IsotonicProbabilityCalibrationModel& calibrationModel, const CContiguousLabelMatrix& labelMatrix,
      const CContiguousConstView<float64>& scoreMatrix,
      const IMarginalProbabilityFunction& marginalProbabilityFunction) {
        for (uint32 i = 0; i < numExamples; i++) {
            uint32 exampleIndex = indexIterator[i];
            CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.values_cbegin(exampleIndex);
            CContiguousConstView<float64>::value_const_iterator scoreIterator = scoreMatrix.values_cbegin(exampleIndex);

            for (uint32 j = 0; j < numLabels; j++) {
                float64 trueProbability = labelIterator[j] ? 1 : 0;
                float64 score = scoreIterator[j];
                float64 marginalProbability =
                  marginalProbabilityFunction.transformScoreIntoMarginalProbability(j, score);
                calibrationModel.addBin(j, marginalProbability, trueProbability);
            }
        }
    }

    template<typename IndexIterator>
    static inline void extractThresholdsAndProbabilities(
      IndexIterator indexIterator, uint32 numExamples, uint32 numLabels,
      IsotonicProbabilityCalibrationModel& calibrationModel, const CsrLabelMatrix& labelMatrix,
      const CContiguousConstView<float64>& scoreMatrix,
      const IMarginalProbabilityFunction& marginalProbabilityFunction) {
        for (uint32 i = 0; i < numExamples; i++) {
            uint32 exampleIndex = indexIterator[i];
            auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                              labelMatrix.indices_cend(exampleIndex));
            CContiguousConstView<float64>::value_const_iterator scoreIterator = scoreMatrix.values_cbegin(exampleIndex);

            for (uint32 j = 0; j < numLabels; j++) {
                float64 trueProbability = (*labelIterator) ? 1 : 0;
                float64 score = scoreIterator[j];
                float64 marginalProbability =
                  marginalProbabilityFunction.transformScoreIntoMarginalProbability(j, score);
                calibrationModel.addBin(j, marginalProbability, trueProbability);
                labelIterator++;
            }
        }
    }

    template<typename IndexIterator>
    static inline void extractThresholdsAndProbabilities(
      IndexIterator indexIterator, uint32 numExamples, uint32 numLabels,
      IsotonicProbabilityCalibrationModel& calibrationModel, const CContiguousLabelMatrix& labelMatrix,
      const SparseSetMatrix<float64>& scoreMatrix, const IMarginalProbabilityFunction& marginalProbabilityFunction) {
        for (uint32 i = 0; i < numLabels; i++) {
            calibrationModel.addBin(i, 0, 0);
        }

        uint32* numSparsePerLabel = new uint32[numLabels] {};

        for (uint32 i = 0; i < numExamples; i++) {
            uint32 exampleIndex = indexIterator[i];
            CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.values_cbegin(exampleIndex);
            SparseSetMatrix<float64>::const_row scoreRow = scoreMatrix[exampleIndex];

            for (uint32 j = 0; j < numLabels; j++) {
                float64 trueProbability = labelIterator[j] ? 1 : 0;
                const IndexedValue<float64>* entry = scoreRow[j];

                if (entry) {
                    float64 score = entry->value;
                    float64 marginalProbability =
                      marginalProbabilityFunction.transformScoreIntoMarginalProbability(j, score);
                    calibrationModel.addBin(j, marginalProbability, trueProbability);
                } else {
                    IsotonicProbabilityCalibrationModel::bin_list bins = calibrationModel[j];
                    Tuple<float64>& firstBin = bins[0];
                    uint32 numSparse = numSparsePerLabel[j] + 1;

                    if (numSparse > 1) {
                        firstBin.second = iterativeArithmeticMean(numSparse, trueProbability, firstBin.second);
                    } else {
                        firstBin.second = trueProbability;
                    }

                    numSparsePerLabel[j] = numSparse;
                }
            }
        }

        delete[] numSparsePerLabel;
    }

    template<typename IndexIterator>
    static inline void extractThresholdsAndProbabilities(
      IndexIterator indexIterator, uint32 numExamples, uint32 numLabels,
      IsotonicProbabilityCalibrationModel& calibrationModel, const CsrLabelMatrix& labelMatrix,
      const SparseSetMatrix<float64>& scoreMatrix, const IMarginalProbabilityFunction& marginalProbabilityFunction) {
        for (uint32 i = 0; i < numLabels; i++) {
            calibrationModel.addBin(i, 0, 0);
        }

        uint32* numSparsePerLabel = new uint32[numLabels];
        setArrayToValue(numSparsePerLabel, numLabels, numExamples);
        uint32* numSparseRelevantPerLabel = new uint32[numLabels] {};

        for (uint32 i = 0; i < numExamples; i++) {
            uint32 exampleIndex = indexIterator[i];
            CsrLabelMatrix::index_const_iterator labelIndicesBegin = labelMatrix.indices_cbegin(exampleIndex);
            CsrLabelMatrix::index_const_iterator labelIndicesEnd = labelMatrix.indices_cend(exampleIndex);
            uint32 numRelevantLabels = labelIndicesEnd - labelIndicesBegin;

            for (uint32 j = 0; j < numRelevantLabels; j++) {
                uint32 labelIndex = labelIndicesBegin[j];
                numSparseRelevantPerLabel[labelIndex] += 1;
            }

            for (auto it = scoreMatrix.cbegin(exampleIndex); it != scoreMatrix.cend(exampleIndex); it++) {
                const IndexedValue<float64>& entry = *it;
                uint32 labelIndex = entry.index;
                float64 score = entry.value;
                float64 marginalProbability =
                  marginalProbabilityFunction.transformScoreIntoMarginalProbability(labelIndex, score);
                bool trueLabel = std::binary_search(labelIndicesBegin, labelIndicesEnd, labelIndex);
                calibrationModel.addBin(labelIndex, marginalProbability, trueLabel ? 1 : 0);
                numSparsePerLabel[labelIndex] -= 1;

                if (trueLabel) {
                    numSparseRelevantPerLabel[labelIndex] -= 1;
                }
            }
        }

        for (uint32 i = 0; i < numLabels; i++) {
            IsotonicProbabilityCalibrationModel::bin_list bins = calibrationModel[i];
            Tuple<float64>& firstBin = bins[0];
            firstBin.second = (float64) numSparseRelevantPerLabel[i] / (float64) numSparsePerLabel[i];
        }

        delete[] numSparsePerLabel;
        delete[] numSparseRelevantPerLabel;
    }

    template<typename IndexIterator, typename LabelMatrix>
    static inline std::unique_ptr<IsotonicProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
      IndexIterator indexIterator, uint32 numExamples, const LabelMatrix& labelMatrix, const IStatistics& statistics,
      const IMarginalProbabilityFunction& marginalProbabilityFunction) {
        // Extract thresholds and ground truth probabilities from score matrix and label matrix, respectively...
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<IsotonicProbabilityCalibrationModel> calibrationModelPtr =
          std::make_unique<IsotonicProbabilityCalibrationModel>(numLabels);
        const IBoostingStatistics& boostingStatistics = dynamic_cast<const IBoostingStatistics&>(statistics);
        auto denseVisitor =
          [=, &marginalProbabilityFunction, &calibrationModelPtr](const CContiguousConstView<float64>& scoreMatrix) {
            extractThresholdsAndProbabilities(indexIterator, numExamples, numLabels, *calibrationModelPtr, labelMatrix,
                                              scoreMatrix, marginalProbabilityFunction);
        };
        auto sparseVisitor =
          [=, &marginalProbabilityFunction, &calibrationModelPtr](const SparseSetMatrix<float64>& scoreMatrix) {
            extractThresholdsAndProbabilities(indexIterator, numExamples, numLabels, *calibrationModelPtr, labelMatrix,
                                              scoreMatrix, marginalProbabilityFunction);
        };
        boostingStatistics.visitScoreMatrix(denseVisitor, sparseVisitor);

        // Build and return the isotonic calibration model...
        calibrationModelPtr->fit();
        return calibrationModelPtr;
    }

    template<typename LabelMatrix>
    static inline std::unique_ptr<IsotonicProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
      const SinglePartition& partition, const LabelMatrix& labelMatrix, const IStatistics& statistics,
      const IMarginalProbabilityFunction& marginalProbabilityFunction) {
        return fitMarginalProbabilityCalibrationModel(partition.cbegin(), partition.getNumElements(), labelMatrix,
                                                      statistics, marginalProbabilityFunction);
    }

    template<typename LabelMatrix>
    static inline std::unique_ptr<IsotonicProbabilityCalibrationModel> fitMarginalProbabilityCalibrationModel(
      const BiPartition& partition, uint32 useHoldoutSet, const LabelMatrix& labelMatrix, const IStatistics& statistics,
      const IMarginalProbabilityFunction& marginalProbabilityFunction) {
        BiPartition::const_iterator indexIterator;
        uint32 numExamples;

        if (useHoldoutSet) {
            indexIterator = partition.second_cbegin();
            numExamples = partition.getNumSecond();
        } else {
            indexIterator = partition.first_cbegin();
            numExamples = partition.getNumFirst();
        }

        return fitMarginalProbabilityCalibrationModel(indexIterator, numExamples, labelMatrix, statistics,
                                                      marginalProbabilityFunction);
    }

    /**
     * An implementation of the type `IMarginalProbabilityCalibrator` that does fit a model for the calibration of
     * marginal probabilities via isotonic regression.
     */
    class IsotonicMarginalProbabilityCalibrator final : public IMarginalProbabilityCalibrator {
        private:

            const std::unique_ptr<IMarginalProbabilityCalibrationModel> marginalProbabilityCalibrationModelPtr_;

            const std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr_;

            const bool useHoldoutSet_;

        public:

            /**
             * @param marginalProbabilityFunctionFactory  A reference to an object of type
             *                                            `IMarginalProbabilityFunctionFactory` that allows to create
             *                                            implementations of the transformation function to be used to
             *                                            transform regression scores that are predicted for
             *                                            individual labels into marginal probabilities
             * @param useHoldoutSet                       True, if the calibration model should be fit to the examples
             *                                            in the holdout set, if available, false otherwise
             */
            IsotonicMarginalProbabilityCalibrator(
              const IMarginalProbabilityFunctionFactory& marginalProbabilityFunctionFactory, bool useHoldoutSet)
                : marginalProbabilityCalibrationModelPtr_(createNoProbabilityCalibrationModel()),
                  marginalProbabilityFunctionPtr_(
                    marginalProbabilityFunctionFactory.create(*marginalProbabilityCalibrationModelPtr_)),
                  useHoldoutSet_(useHoldoutSet) {}

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return fitMarginalProbabilityCalibrationModel(partition, labelMatrix, statistics,
                                                              *marginalProbabilityFunctionPtr_);
            }

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              const SinglePartition& partition, const CsrLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return fitMarginalProbabilityCalibrationModel(partition, labelMatrix, statistics,
                                                              *marginalProbabilityFunctionPtr_);
            }

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              BiPartition& partition, const CContiguousLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return fitMarginalProbabilityCalibrationModel(partition, useHoldoutSet_, labelMatrix, statistics,
                                                              *marginalProbabilityFunctionPtr_);
            }

            /**
             * @see `IMarginalProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IMarginalProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics) const override {
                return fitMarginalProbabilityCalibrationModel(partition, useHoldoutSet_, labelMatrix, statistics,
                                                              *marginalProbabilityFunctionPtr_);
            }
    };

    /**
     * A factory that allows to create instances of the type `IsotonicMarginalProbabilityCalibrator`.
     */
    class IsotonicMarginalProbabilityCalibratorFactory final : public IMarginalProbabilityCalibratorFactory {
        private:

            const std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr_;

            const bool useHoldoutSet_;

        public:

            /**
             * @param marginalProbabilityFunctionFactoryPtr An unique pointer to an object of type
             *                                              `IMarginalProbabilityFunctionFactory` that allows to create
             *                                              implementations of the transformation function to be used to
             *                                              transform regression scores that are predicted for
             *                                              individual labels into marginal probabilities
             * @param useHoldoutSet                         True, if the calibration model should be fit to the examples
             *                                              in the holdout set, if available, false otherwise
             */
            IsotonicMarginalProbabilityCalibratorFactory(
              std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr,
              bool useHoldoutSet)
                : marginalProbabilityFunctionFactoryPtr_(std::move(marginalProbabilityFunctionFactoryPtr)),
                  useHoldoutSet_(useHoldoutSet) {}

            /**
             * @see `IMarginalProbabilityCalibratorFactory::create`
             */
            std::unique_ptr<IMarginalProbabilityCalibrator> create() const override {
                return std::make_unique<IsotonicMarginalProbabilityCalibrator>(*marginalProbabilityFunctionFactoryPtr_,
                                                                               useHoldoutSet_);
            }
    };

    IsotonicMarginalProbabilityCalibratorConfig::IsotonicMarginalProbabilityCalibratorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr)
        : useHoldoutSet_(true), lossConfigPtr_(lossConfigPtr) {}

    bool IsotonicMarginalProbabilityCalibratorConfig::isHoldoutSetUsed() const {
        return useHoldoutSet_;
    }

    IIsotonicMarginalProbabilityCalibratorConfig& IsotonicMarginalProbabilityCalibratorConfig::setUseHoldoutSet(
      bool useHoldoutSet) {
        useHoldoutSet_ = useHoldoutSet;
        return *this;
    }

    bool IsotonicMarginalProbabilityCalibratorConfig::shouldUseHoldoutSet() const {
        return useHoldoutSet_;
    }

    std::unique_ptr<IMarginalProbabilityCalibratorFactory>
      IsotonicMarginalProbabilityCalibratorConfig::createMarginalProbabilityCalibratorFactory() const {
        std::unique_ptr<IMarginalProbabilityFunctionFactory> marginalProbabilityFunctionFactoryPtr =
          lossConfigPtr_->createMarginalProbabilityFunctionFactory();

        if (marginalProbabilityFunctionFactoryPtr) {
            return std::make_unique<IsotonicMarginalProbabilityCalibratorFactory>(
              std::move(marginalProbabilityFunctionFactoryPtr), useHoldoutSet_);
        } else {
            return std::make_unique<NoMarginalProbabilityCalibratorFactory>();
        }
    }

    template<typename LabelIndexIterator>
    static inline bool areLabelVectorsEqual(LabelIndexIterator labelIndicesBegin, LabelIndexIterator labelIndicesEnd,
                                            const LabelVector& labelVector) {
        uint32 numRelevantLabels = labelVector.getNumElements();
        LabelVector::const_iterator labelIndexIterator = labelVector.cbegin();

        for (uint32 i = 0; i < numRelevantLabels; i++) {
            if (labelIndicesBegin == labelIndicesEnd || *labelIndicesBegin != labelIndexIterator[i]) {
                return false;
            }

            labelIndicesBegin++;
        }

        return true;
    }

    template<typename IndexIterator>
    static inline void extractThresholdsAndProbabilities(IndexIterator indexIterator, uint32 numExamples,
                                                         IsotonicProbabilityCalibrationModel& calibrationModel,
                                                         const CContiguousLabelMatrix& labelMatrix,
                                                         const CContiguousConstView<float64>& scoreMatrix,
                                                         const IJointProbabilityFunction& jointProbabilityFunction,
                                                         const LabelVectorSet& labelVectorSet) {
        LabelVectorSet::const_iterator labelVectorIterator = labelVectorSet.cbegin();
        uint32 numLabelVectors = labelVectorSet.getNumLabelVectors();

        for (uint32 i = 0; i < numLabelVectors; i++) {
            IsotonicProbabilityCalibrationModel::bin_list bins = calibrationModel[i];
            const LabelVector& labelVector = *labelVectorIterator[i];

            for (uint32 j = 0; j < numExamples; j++) {
                uint32 exampleIndex = indexIterator[j];
                auto labelIndicesBegin = make_non_zero_index_forward_iterator(labelMatrix.values_cbegin(exampleIndex),
                                                                              labelMatrix.values_cend(exampleIndex));
                auto labelIndicesEnd = make_non_zero_index_forward_iterator(labelMatrix.values_cend(exampleIndex),
                                                                            labelMatrix.values_cend(exampleIndex));
                float64 trueProbability = areLabelVectorsEqual(labelIndicesBegin, labelIndicesEnd, labelVector) ? 1 : 0;
                CContiguousConstView<float64>::value_const_iterator scoresBegin =
                  scoreMatrix.values_cbegin(exampleIndex);
                CContiguousConstView<float64>::value_const_iterator scoresEnd = scoreMatrix.values_cend(exampleIndex);
                float64 jointProbability =
                  jointProbabilityFunction.transformScoresIntoJointProbability(i, labelVector, scoresBegin, scoresEnd);
                bins.emplace_back(jointProbability, trueProbability);
            }
        }
    }

    template<typename IndexIterator>
    static inline void extractThresholdsAndProbabilities(IndexIterator indexIterator, uint32 numExamples,
                                                         IsotonicProbabilityCalibrationModel& calibrationModel,
                                                         const CsrLabelMatrix& labelMatrix,
                                                         const CContiguousConstView<float64>& scoreMatrix,
                                                         const IJointProbabilityFunction& jointProbabilityFunction,
                                                         const LabelVectorSet& labelVectorSet) {
        LabelVectorSet::const_iterator labelVectorIterator = labelVectorSet.cbegin();
        uint32 numLabelVectors = labelVectorSet.getNumLabelVectors();

        for (uint32 i = 0; i < numLabelVectors; i++) {
            IsotonicProbabilityCalibrationModel::bin_list bins = calibrationModel[i];
            const LabelVector& labelVector = *labelVectorIterator[i];

            for (uint32 j = 0; j < numExamples; j++) {
                uint32 exampleIndex = indexIterator[j];
                CsrLabelMatrix::index_const_iterator labelIndicesBegin = labelMatrix.indices_cbegin(exampleIndex);
                CsrLabelMatrix::index_const_iterator labelIndicesEnd = labelMatrix.indices_cend(exampleIndex);
                float64 trueProbability = areLabelVectorsEqual(labelIndicesBegin, labelIndicesEnd, labelVector) ? 1 : 0;
                CContiguousConstView<float64>::value_const_iterator scoresBegin =
                  scoreMatrix.values_cbegin(exampleIndex);
                CContiguousConstView<float64>::value_const_iterator scoresEnd = scoreMatrix.values_cend(exampleIndex);
                float64 jointProbability =
                  jointProbabilityFunction.transformScoresIntoJointProbability(i, labelVector, scoresBegin, scoresEnd);
                bins.emplace_back(jointProbability, trueProbability);
            }
        }
    }

    template<typename IndexIterator>
    static inline void extractThresholdsAndProbabilities(IndexIterator indexIterator, uint32 numExamples,
                                                         IsotonicProbabilityCalibrationModel& calibrationModel,
                                                         const CContiguousLabelMatrix& labelMatrix,
                                                         const SparseSetMatrix<float64>& scoreMatrix,
                                                         const IJointProbabilityFunction& jointProbabilityFunction,
                                                         const LabelVectorSet& labelVectorSet) {
        LabelVectorSet::const_iterator labelVectorIterator = labelVectorSet.cbegin();
        uint32 numLabelVectors = labelVectorSet.getNumLabelVectors();
        uint32 numLabels = labelMatrix.getNumCols();

        for (uint32 i = 0; i < numLabelVectors; i++) {
            IsotonicProbabilityCalibrationModel::bin_list bins = calibrationModel[i];
            const LabelVector& labelVector = *labelVectorIterator[i];

            for (uint32 j = 0; j < numExamples; j++) {
                uint32 exampleIndex = indexIterator[j];
                auto labelIndicesBegin = make_non_zero_index_forward_iterator(labelMatrix.values_cbegin(exampleIndex),
                                                                              labelMatrix.values_cend(exampleIndex));
                auto labelIndicesEnd = make_non_zero_index_forward_iterator(labelMatrix.values_cend(exampleIndex),
                                                                            labelMatrix.values_cend(exampleIndex));
                float64 trueProbability = areLabelVectorsEqual(labelIndicesBegin, labelIndicesEnd, labelVector) ? 1 : 0;
                SparseSetMatrix<float64>::const_row scores = scoreMatrix[exampleIndex];
                float64 jointProbability =
                  jointProbabilityFunction.transformScoresIntoJointProbability(i, labelVector, scores, numLabels);
                bins.emplace_back(jointProbability, trueProbability);
            }
        }
    }

    template<typename IndexIterator>
    static inline void extractThresholdsAndProbabilities(IndexIterator indexIterator, uint32 numExamples,
                                                         IsotonicProbabilityCalibrationModel& calibrationModel,
                                                         const CsrLabelMatrix& labelMatrix,
                                                         const SparseSetMatrix<float64>& scoreMatrix,
                                                         const IJointProbabilityFunction& jointProbabilityFunction,
                                                         const LabelVectorSet& labelVectorSet) {
        LabelVectorSet::const_iterator labelVectorIterator = labelVectorSet.cbegin();
        uint32 numLabelVectors = labelVectorSet.getNumLabelVectors();
        uint32 numLabels = labelMatrix.getNumCols();

        for (uint32 i = 0; i < numLabelVectors; i++) {
            IsotonicProbabilityCalibrationModel::bin_list bins = calibrationModel[i];
            const LabelVector& labelVector = *labelVectorIterator[i];

            for (uint32 j = 0; j < numExamples; j++) {
                uint32 exampleIndex = indexIterator[j];
                CsrLabelMatrix::index_const_iterator labelIndicesBegin = labelMatrix.indices_cbegin(exampleIndex);
                CsrLabelMatrix::index_const_iterator labelIndicesEnd = labelMatrix.indices_cend(exampleIndex);
                float64 trueProbability = areLabelVectorsEqual(labelIndicesBegin, labelIndicesEnd, labelVector) ? 1 : 0;
                SparseSetMatrix<float64>::const_row scores = scoreMatrix[exampleIndex];
                float64 jointProbability =
                  jointProbabilityFunction.transformScoresIntoJointProbability(i, labelVector, scores, numLabels);
                bins.emplace_back(jointProbability, trueProbability);
            }
        }
    }

    template<typename IndexIterator, typename LabelMatrix>
    static inline std::unique_ptr<IsotonicProbabilityCalibrationModel> fitJointProbabilityCalibrationModel(
      IndexIterator indexIterator, uint32 numExamples, const LabelMatrix& labelMatrix, const IStatistics& statistics,
      const IJointProbabilityFunction& jointProbabilityFunction, const LabelVectorSet& labelVectorSet) {
        // Extract thresholds and ground truth probabilities from score matrix and label matrix, respectively...
        uint32 numLabelVectors = labelVectorSet.getNumLabelVectors();
        std::unique_ptr<IsotonicProbabilityCalibrationModel> calibrationModelPtr =
          std::make_unique<IsotonicProbabilityCalibrationModel>(numLabelVectors);
        const IBoostingStatistics& boostingStatistics = dynamic_cast<const IBoostingStatistics&>(statistics);
        auto denseVisitor = [=, &jointProbabilityFunction, &calibrationModelPtr,
                             &labelVectorSet](const CContiguousConstView<float64>& scoreMatrix) {
            extractThresholdsAndProbabilities(indexIterator, numExamples, *calibrationModelPtr, labelMatrix,
                                              scoreMatrix, jointProbabilityFunction, labelVectorSet);
        };
        auto sparseVisitor = [=, &jointProbabilityFunction, &calibrationModelPtr,
                              &labelVectorSet](const SparseSetMatrix<float64>& scoreMatrix) {
            extractThresholdsAndProbabilities(indexIterator, numExamples, *calibrationModelPtr, labelMatrix,
                                              scoreMatrix, jointProbabilityFunction, labelVectorSet);
        };
        boostingStatistics.visitScoreMatrix(denseVisitor, sparseVisitor);

        // Build and return the isotonic calibration model...
        calibrationModelPtr->fit();
        return calibrationModelPtr;
    }

    template<typename LabelMatrix>
    static inline std::unique_ptr<IsotonicProbabilityCalibrationModel> fitJointProbabilityCalibrationModel(
      const SinglePartition& partition, const LabelMatrix& labelMatrix, const IStatistics& statistics,
      const IJointProbabilityFunction& jointProbabilityFunction, const LabelVectorSet& labelVectorSet) {
        return fitJointProbabilityCalibrationModel(partition.cbegin(), partition.getNumElements(), labelMatrix,
                                                   statistics, jointProbabilityFunction, labelVectorSet);
    }

    template<typename LabelMatrix>
    static inline std::unique_ptr<IsotonicProbabilityCalibrationModel> fitJointProbabilityCalibrationModel(
      const BiPartition& partition, bool useHoldoutSet, const LabelMatrix& labelMatrix, const IStatistics& statistics,
      const IJointProbabilityFunction& jointProbabilityFunction, const LabelVectorSet& labelVectorSet) {
        BiPartition::const_iterator indexIterator;
        uint32 numExamples;

        if (useHoldoutSet) {
            indexIterator = partition.second_cbegin();
            numExamples = partition.getNumSecond();
        } else {
            indexIterator = partition.first_cbegin();
            numExamples = partition.getNumFirst();
        }

        return fitJointProbabilityCalibrationModel(indexIterator, numExamples, labelMatrix, statistics,
                                                   jointProbabilityFunction, labelVectorSet);
    }

    /**
     * An implementation of the type `IJointProbabilityCalibrator` that does fit a model for the calibration of joint
     * probabilities via isotonic regression.
     */
    class IsotonicJointProbabilityCalibrator final : public IJointProbabilityCalibrator {
        private:

            const std::unique_ptr<IJointProbabilityCalibrationModel> jointProbabilityCalibrationModelPtr_;

            const std::unique_ptr<IJointProbabilityFunction> jointProbabilityFunctionPtr_;

            const bool useHoldoutSet_;

            const LabelVectorSet& labelVectorSet_;

        public:

            /**
             * @param marginalProbabilityCalibrationModel A reference to an object of type
             *                                            `IMarginalProbabilityCalibrationModel` that may be used for
             *                                            the calibration of marginal probabilities
             * @param jointProbabilityFunctionFactory     A reference to an object of type
             *                                            `IJointProbabilityFunctionFactory` that allows to create
             *                                            implementations of the transformation function to be used to
             *                                            transform regression scores that are predicted for individual
             *                                            labels into marginal probabilities
             * @param useHoldoutSet                       True, if the calibration model should be fit to the examples
             *                                            in the holdout set, if available, false otherwise
             * @param labelVectorSet                      A reference to an object of type `LabelVectorSet` that stores
             *                                            all known label vectors
             */
            IsotonicJointProbabilityCalibrator(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityFunctionFactory& jointProbabilityFunctionFactory, bool useHoldoutSet,
              const LabelVectorSet& labelVectorSet)
                : jointProbabilityCalibrationModelPtr_(createNoProbabilityCalibrationModel()),
                  jointProbabilityFunctionPtr_(jointProbabilityFunctionFactory.create(
                    marginalProbabilityCalibrationModel, *jointProbabilityCalibrationModelPtr_)),
                  useHoldoutSet_(useHoldoutSet), labelVectorSet_(labelVectorSet) {}

            /**
             * @see `IJointProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              const SinglePartition& partition, const CContiguousLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return fitJointProbabilityCalibrationModel(partition, labelMatrix, statistics,
                                                           *jointProbabilityFunctionPtr_, labelVectorSet_);
            }

            /**
             * @see `IJointProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              const SinglePartition& partition, const CsrLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return fitJointProbabilityCalibrationModel(partition, labelMatrix, statistics,
                                                           *jointProbabilityFunctionPtr_, labelVectorSet_);
            }

            /**
             * @see `IJointProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              BiPartition& partition, const CContiguousLabelMatrix& labelMatrix,
              const IStatistics& statistics) const override {
                return fitJointProbabilityCalibrationModel(partition, useHoldoutSet_, labelMatrix, statistics,
                                                           *jointProbabilityFunctionPtr_, labelVectorSet_);
            }

            /**
             * @see `IJointProbabilityCalibrator::fitProbabilityCalibrationModel`
             */
            std::unique_ptr<IJointProbabilityCalibrationModel> fitProbabilityCalibrationModel(
              BiPartition& partition, const CsrLabelMatrix& labelMatrix, const IStatistics& statistics) const override {
                return fitJointProbabilityCalibrationModel(partition, useHoldoutSet_, labelMatrix, statistics,
                                                           *jointProbabilityFunctionPtr_, labelVectorSet_);
            }
    };

    /**
     * A factory that allows to create instances of the type `IsotonicJointProbabilityCalibrator`.
     */
    class IsotonicJointProbabilityCalibratorFactory final : public IJointProbabilityCalibratorFactory {
        private:

            const std::unique_ptr<IJointProbabilityFunctionFactory> jointProbabilityFunctionFactoryPtr_;

            const bool useHoldoutSet_;

        public:

            /**
             * @param jointProbabilityFunctionFactoryPtr  An unique pointer to an object of type
             *                                            `IJointProbabilityFunctionFactory` that allows to create
             *                                            implementations of the transformation function to be used to
             *                                            transform regression scores that are predicted for individual
             *                                            labels into joint probabilities
             * @param useHoldoutSet                       True, if the calibration model should be fit to the examples
             *                                            in the holdout set, if available, false otherwise
             */
            IsotonicJointProbabilityCalibratorFactory(
              std::unique_ptr<IJointProbabilityFunctionFactory> jointProbabilityFunctionFactoryPtr, bool useHoldoutSet)
                : jointProbabilityFunctionFactoryPtr_(std::move(jointProbabilityFunctionFactoryPtr)),
                  useHoldoutSet_(useHoldoutSet) {}

            /**
             * @see `IJointProbabilityCalibratorFactory::create`
             */
            std::unique_ptr<IJointProbabilityCalibrator> create(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const LabelVectorSet* labelVectorSet) const override {
                if (!labelVectorSet) {
                    throw std::runtime_error(
                      "Information about the label vectors that have been encountered in the training data is required "
                      "for fitting a model for the calibration of joint probabilities, but no such information is "
                      "provided by the model. Most probably, the model was intended to use a different calibration "
                      "method when it has been trained.");
                }

                return std::make_unique<IsotonicJointProbabilityCalibrator>(marginalProbabilityCalibrationModel,
                                                                            *jointProbabilityFunctionFactoryPtr_,
                                                                            useHoldoutSet_, *labelVectorSet);
            }
    };

    IsotonicJointProbabilityCalibratorConfig::IsotonicJointProbabilityCalibratorConfig(
      const std::unique_ptr<ILossConfig>& lossConfigPtr)
        : useHoldoutSet_(true), lossConfigPtr_(lossConfigPtr) {}

    bool IsotonicJointProbabilityCalibratorConfig::isHoldoutSetUsed() const {
        return useHoldoutSet_;
    }

    IIsotonicJointProbabilityCalibratorConfig& IsotonicJointProbabilityCalibratorConfig::setUseHoldoutSet(
      bool useHoldoutSet) {
        useHoldoutSet_ = useHoldoutSet;
        return *this;
    }

    bool IsotonicJointProbabilityCalibratorConfig::shouldUseHoldoutSet() const {
        return useHoldoutSet_;
    }

    bool IsotonicJointProbabilityCalibratorConfig::isLabelVectorSetNeeded() const {
        return true;
    }

    std::unique_ptr<IJointProbabilityCalibratorFactory>
      IsotonicJointProbabilityCalibratorConfig::createJointProbabilityCalibratorFactory() const {
        std::unique_ptr<IJointProbabilityFunctionFactory> jointProbabilityFunctionFactoryPtr =
          lossConfigPtr_->createJointProbabilityFunctionFactory();

        if (jointProbabilityFunctionFactoryPtr) {
            return std::make_unique<IsotonicJointProbabilityCalibratorFactory>(
              std::move(jointProbabilityFunctionFactoryPtr), useHoldoutSet_);
        } else {
            return std::make_unique<NoJointProbabilityCalibratorFactory>();
        }
    }
}
