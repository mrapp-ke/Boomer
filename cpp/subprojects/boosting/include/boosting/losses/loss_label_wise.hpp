/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/measures/measure_evaluation.hpp"
#include "common/measures/measure_similarity.hpp"
#include "boosting/data/statistic_view_label_wise_dense.hpp"


namespace boosting {

    /**
     * Defines an interface for all (decomposable) loss functions that are applied label-wise.
     */
    class ILabelWiseLoss : public IEvaluationMeasure, public ISimilarityMeasure {

        public:

            virtual ~ILabelWiseLoss() { };

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousLabelMatrix` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the
             *                          currently predicted scores
             * @param labelIndicesBegin A `CompleteIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `CompleteIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `DenseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CContiguousLabelMatrix` that provides random
             *                          access to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the
             *                          currently predicted scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `DenseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `CompleteIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CsrLabelMatrix` that provides row-wise access
             *                          to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the
             *                          currently predicted scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `DenseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index, considering only the labels, whose indices are
             * provided by a `PartialIndexVector`.
             *
             * @param exampleIndex      The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix       A reference to an object of type `CsrLabelMatrix` that provides row-wise access
             *                          to the labels of the training examples
             * @param scoreMatrix       A reference to an object of type `CContiguousConstView` that stores the
             *                          currently predicted scores
             * @param labelIndicesBegin A `PartialIndexVector::const_iterator` to the beginning of the label indices
             * @param labelIndicesEnd   A `PartialIndexVector::const_iterator` to the end of the label indices
             * @param statisticView     A reference to an object of type `DenseLabelWiseStatisticView` to be updated
             */
            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                                   const CContiguousConstView<float64> scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const = 0;

    };

    /**
     * An abstract base class for all (decomposable) loss functions that are applied label-wise.
     */
    class AbstractLabelWiseLoss : public ILabelWiseLoss {

        private:

            /**
             * A function that allows to update the gradient and Hessian for a single example and label. The function
             * accepts the true label, the predicted score, as well as pointers to the gradient and Hessian to be
             * updated, as arguments.
             */
            typedef void (*UpdateFunction)(bool trueLabel, float64 predictedScore, float64* gradient, float64* hessian);

            /**
             * A function that allows to calculate a numerical score that assesses the quality of the prediction for a
             * single example and label. The function accepts the true label and the predicted score as arguments and
             * returns a numerical score.
             */
            typedef float64 (*EvaluateFunction)(bool trueLabel, float64 predictedScore);

            UpdateFunction updateFunction_;

            EvaluateFunction evaluateFunction_;

        protected:

            /**
             * @param updateFunction    The function to be used for updating gradients and Hessians
             * @param evaluateFunction  The function to be used for evaluating predictions
             */
            AbstractLabelWiseLoss(UpdateFunction updateFunction, EvaluateFunction evaluateFunction);

        public:

            virtual ~AbstractLabelWiseLoss() { };

            void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                           const CContiguousConstView<float64>& scoreMatrix,
                                           CompleteIndexVector::const_iterator labelIndicesBegin,
                                           CompleteIndexVector::const_iterator labelIndicesEnd,
                                           DenseLabelWiseStatisticView& statisticView) const override final;

            void updateLabelWiseStatistics(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                                           const CContiguousConstView<float64>& scoreMatrix,
                                           PartialIndexVector::const_iterator labelIndicesBegin,
                                           PartialIndexVector::const_iterator labelIndicesEnd,
                                           DenseLabelWiseStatisticView& statisticView) const override final;

            void updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                           const CContiguousConstView<float64>& scoreMatrix,
                                           CompleteIndexVector::const_iterator labelIndicesBegin,
                                           CompleteIndexVector::const_iterator labelIndicesEnd,
                                           DenseLabelWiseStatisticView& statisticView) const override final;

            void updateLabelWiseStatistics(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                                            const CContiguousConstView<float64> scoreMatrix,
                                            PartialIndexVector::const_iterator labelIndicesBegin,
                                            PartialIndexVector::const_iterator labelIndicesEnd,
                                            DenseLabelWiseStatisticView& statisticView) const override final;

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                             const CContiguousConstView<float64>& scoreMatrix) const override final;

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                             const CContiguousConstView<float64>& scoreMatrix) const override final;

            /**
             * @see `ISimilarityMeasure::measureSimilarity`
             */
            float64 measureSimilarity(const LabelVector& labelVector,
                                      CContiguousView<float64>::const_iterator scoresBegin,
                                      CContiguousView<float64>::const_iterator scoresEnd) const override final;

    };

}
