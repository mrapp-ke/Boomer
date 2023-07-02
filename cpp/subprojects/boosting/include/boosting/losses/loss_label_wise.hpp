/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/data/statistic_view_label_wise_dense.hpp"
#include "boosting/losses/loss.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"

namespace boosting {

    /**
     * Defines an interface for all (decomposable) loss functions that are applied label-wise.
     */
    class ILabelWiseLoss : public ILoss {
        public:

            virtual ~ILabelWiseLoss() override {};

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousConstView` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the
             *                          currently predicted scores
             * @param labelIndicesBegin A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `DenseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex,
                                                   const CContiguousConstView<const uint8>& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousConstView` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the
             *                          currently predicted scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `DenseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex,
                                                   const CContiguousConstView<const uint8>& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `BinaryCsrConstView` that provides row-wise
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the
             *                          currently predicted scores
             * @param labelIndicesBegin A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `DenseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `BinaryCsrConstView` that provides row-wise
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the
             *                          currently predicted scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `DenseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `ILabelWiseLoss`.
     */
    class ILabelWiseLossFactory : public IEvaluationMeasureFactory,
                                  public IDistanceMeasureFactory {
        public:

            virtual ~ILabelWiseLossFactory() override {};

            /**
             * Creates and returns a new object of type `ILabelWiseLoss`.
             *
             * @return An unique pointer to an object of type `ILabelWiseLoss` that has been created
             */
            virtual std::unique_ptr<ILabelWiseLoss> createLabelWiseLoss() const = 0;

            /**
             * @see `IEvaluationMeasureFactory::createEvaluationMeasure`
             */
            std::unique_ptr<IEvaluationMeasure> createEvaluationMeasure() const override final {
                return this->createLabelWiseLoss();
            }

            /**
             * @see `IDistanceMeasureFactory::createDistanceMeasure`
             */
            std::unique_ptr<IDistanceMeasure> createDistanceMeasure(
              const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
              const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel) const override final {
                return this->createLabelWiseLoss();
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a (decomposable) loss function that is applied
     * label-wise.
     */
    class ILabelWiseLossConfig : public ILossConfig {
        public:

            virtual ~ILabelWiseLossConfig() override {};

            /**
             * Creates and returns a new object of type `ILabelWiseLossFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `ILabelWiseLossFactory` that has been created
             */
            virtual std::unique_ptr<ILabelWiseLossFactory> createLabelWiseLossFactory() const = 0;

            std::unique_ptr<IEvaluationMeasureFactory> createEvaluationMeasureFactory() const override final {
                return this->createLabelWiseLossFactory();
            }

            std::unique_ptr<IDistanceMeasureFactory> createDistanceMeasureFactory() const override final {
                return this->createLabelWiseLossFactory();
            }

            bool isDecomposable() const override final {
                return true;
            }

            bool isSparse() const override {
                return false;
            }
    };

}
