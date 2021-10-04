/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_evaluation/score_vector.hpp"
#include "seco/data/vector_confusion_matrix_dense.hpp"


namespace seco {

    /**
     * Defines an interface for all classes that allow to calculate the predictions of rules, as well as corresponding
     * quality scores, based on confusion matrices.
     */
    class IRuleEvaluation {

        public:

            virtual ~IRuleEvaluation() { };

            /**
             * Calculates the scores to be predicted by a rule, as well as an overall quality score, based on label-wise
             * confusion matrices.
             *
             * @param majorityLabelVector       A reference to an object of type `DenseVector` that stores the
             *                                  predictions of the default rule
             * @param confusionMatricesTotal    A reference to an object of type `DenseConfusionMatrixVector` that
             *                                  stores confusion matrices that take into account all examples
             * @param confusionMatricesCovered  A reference to an object of type `DenseConfusionMatrixVector` that
             *                                  stores confusion matrices that take into account all examples, which are
             *                                  covered by the rule
             * @return                          A reference to an object of type `IScoreVector` that stores the
             *                                  predicted scores, as well as an overall quality score
             */
            virtual const IScoreVector& calculatePrediction(
                const BinarySparseArrayVector& majorityLabelVector,
                const DenseConfusionMatrixVector& confusionMatricesTotal,
                const DenseConfusionMatrixVector& confusionMatricesCovered) = 0;

    };

}
