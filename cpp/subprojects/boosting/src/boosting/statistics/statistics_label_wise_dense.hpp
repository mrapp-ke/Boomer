/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/data/matrix_c_contiguous_numeric.hpp"
#include "boosting/data/statistic_vector_label_wise_dense.hpp"
#include "boosting/data/statistic_view_label_wise_dense.hpp"
#include "boosting/losses/loss_label_wise.hpp"
#include "common/measures/measure_evaluation.hpp"
#include "statistics_label_wise_common.hpp"

#include <cstdlib>

namespace boosting {

    /**
     * A matrix that stores gradients and Hessians that have been calculated using a label-wise decomposable loss
     * function using C-contiguous arrays.
     */
    class DenseLabelWiseStatisticMatrix final : public DenseLabelWiseStatisticView {
        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            DenseLabelWiseStatisticMatrix(uint32 numRows, uint32 numCols)
                : DenseLabelWiseStatisticView(numRows, numCols,
                                              (Tuple<float64>*) malloc(numRows * numCols * sizeof(Tuple<float64>))) {}

            ~DenseLabelWiseStatisticMatrix() override {
                free(statistics_);
            }
    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a differentiable loss function
     * that is applied label-wise and are stored using dense data structures.
     *
     * @tparam LabelMatrix The type of the matrix that provides access to the labels of the training examples
     */
    template<typename LabelMatrix>
    class DenseLabelWiseStatistics final
        : public AbstractLabelWiseStatistics<LabelMatrix, DenseLabelWiseStatisticVector, DenseLabelWiseStatisticView,
                                             DenseLabelWiseStatisticMatrix, NumericCContiguousMatrix<float64>,
                                             ILabelWiseLoss, IEvaluationMeasure, ILabelWiseRuleEvaluationFactory> {
        public:

            /**
             * @param lossPtr               An unique pointer to an object of type `ILabelWiseLoss` that implements the
             *                              loss function that should be used for calculating gradients and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of type `IEvaluationMeasure` that implements
             *                              the evaluation measure that should be used to assess the quality of
             *                              predictions for a specific statistic
             * @param ruleEvaluationFactory A reference to an object of type `ILabelWiseRuleEvaluationFactory`, that
             *                              allows to create instances of the class that is used for calculating the
             *                              predictions of rules, as well as their overall quality
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param statisticViewPtr      An unique pointer to an object of type `DenseLabelWiseStatisticView` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of type `NumericCContiguousMatrix` that
             *                              stores the currently predicted scores
             */
            DenseLabelWiseStatistics(std::unique_ptr<ILabelWiseLoss> lossPtr,
                                     std::unique_ptr<IEvaluationMeasure> evaluationMeasurePtr,
                                     const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                     const LabelMatrix& labelMatrix,
                                     std::unique_ptr<DenseLabelWiseStatisticView> statisticViewPtr,
                                     std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr)
                : AbstractLabelWiseStatistics<LabelMatrix, DenseLabelWiseStatisticVector, DenseLabelWiseStatisticView,
                                              DenseLabelWiseStatisticMatrix, NumericCContiguousMatrix<float64>,
                                              ILabelWiseLoss, IEvaluationMeasure, ILabelWiseRuleEvaluationFactory>(
                  std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, labelMatrix,
                  std::move(statisticViewPtr), std::move(scoreMatrixPtr)) {}

            /**
             * @see `IBoostingStatistics::visitScoreMatrix`
             */
            void visitScoreMatrix(IBoostingStatistics::DenseScoreMatrixVisitor denseVisitor,
                                  IBoostingStatistics::SparseScoreMatrixVisitor sparseVisitor) const override {
                denseVisitor(*this->scoreMatrixPtr_);
            }
    };

}
