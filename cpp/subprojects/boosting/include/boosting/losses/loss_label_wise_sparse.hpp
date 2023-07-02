/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/data/statistic_view_label_wise_sparse.hpp"
#include "boosting/losses/loss_label_wise.hpp"
#include "common/measures/measure_evaluation_sparse.hpp"

namespace boosting {

    /**
     * Defines an interface for all (decomposable) loss functions that are applied label-wise and are suited for the use
     * of sparse data structures. To meet this requirement, the gradients and Hessians that are computed by the loss
     * function should be zero, if the prediction for a label is correct.
     */
    class ISparseLabelWiseLoss : virtual public ILabelWiseLoss,
                                 public ISparseEvaluationMeasure {
        public:

            virtual ~ISparseLabelWiseLoss() override {};

            // Keep "updateLabelWiseStatistics" functions from the parent class rather than hiding them
            using ILabelWiseLoss::updateLabelWiseStatistics;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousConstView` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `SparseSetMatrix` that stores the currently
             *                          predicted scores
             * @param labelIndicesBegin A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex,
                                                   const CContiguousConstView<const uint8>& labelMatrix,
                                                   const SparseSetMatrix<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   SparseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousConstView` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `SparseSetMatrix` that stores the currently
             *                          predicted scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex,
                                                   const CContiguousConstView<const uint8>& labelMatrix,
                                                   const SparseSetMatrix<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   SparseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `BinaryCsrConstView` that provides row-wise
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `SparseSetMatrix` that stores the currently
             *                          predicted scores
             * @param labelIndicesBegin A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                   const SparseSetMatrix<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   SparseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `BinaryCsrConstView` that provides row-wise
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `SparseSetMatrix` that stores the currently
             *                          predicted scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `SparseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                   const SparseSetMatrix<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   SparseLabelWiseStatisticView& statisticView) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `ISparseLabelWiseLoss`.
     */
    class ISparseLabelWiseLossFactory : public ILabelWiseLossFactory,
                                        public ISparseEvaluationMeasureFactory {
        public:

            virtual ~ISparseLabelWiseLossFactory() override {};

            /**
             * Creates and returns a new object of type `ISparseLabelWiseLoss`.
             *
             * @return An unique pointer to an object of type `ISparseLabelWiseLoss` that has been created
             */
            virtual std::unique_ptr<ISparseLabelWiseLoss> createSparseLabelWiseLoss() const = 0;

            /**
             * @see `ILabelWiseLossFactory::createLabelWiseLoss`
             */
            std::unique_ptr<ILabelWiseLoss> createLabelWiseLoss() const override final {
                return this->createSparseLabelWiseLoss();
            }

            /**
             * @see `ISparseEvaluationMeasureFactory::createSparseEvaluationMeasure`
             */
            std::unique_ptr<ISparseEvaluationMeasure> createSparseEvaluationMeasure() const override final {
                return this->createSparseLabelWiseLoss();
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a (decomposable) loss function that is applied
     * label-wise and is suited for the use of sparse data structures.
     */
    class ISparseLabelWiseLossConfig : public ILabelWiseLossConfig {
        public:

            virtual ~ISparseLabelWiseLossConfig() override {};

            /**
             * Creates and returns a new object of type `ISparseLabelWiseLossFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `ISparseLabelWiseLossFactory` that has been created
             */
            virtual std::unique_ptr<ISparseLabelWiseLossFactory> createSparseLabelWiseLossFactory() const = 0;

            /**
             * Creates and returns a new object of type `ISparseEvaluationMeasureFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `ISparseEvaluationMeasureFactory` that has been created
             */
            std::unique_ptr<ISparseEvaluationMeasureFactory> createSparseEvaluationMeasureFactory() const {
                return this->createSparseLabelWiseLossFactory();
            }

            std::unique_ptr<ILabelWiseLossFactory> createLabelWiseLossFactory() const override final {
                return this->createSparseLabelWiseLossFactory();
            }

            bool isSparse() const override final {
                return true;
            }
    };

}
