/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/data/statistic_vector_example_wise_dense.hpp"
#include "rule_evaluation_label_wise_partial_fixed_common.hpp"

namespace boosting {

    /**
     * Calculates scores that assess the quality of optimal predictions for each label and sorts them, such that the
     * first `numPredictions` elements are the best-rated ones.
     *
     * @param tmpIterator               An iterator that provides random access to a temporary array, which should be
     *                                  used to store the sorted scores and their original indices
     * @param gradientIterator          An iterator that provides access to the gradient for each label
     * @param hessianIterator           An iterator that provides access to the Hessian for each label
     * @param numLabels                 The total number of available labels
     * @param numPrediction             The number of the best-rated predictions to be determined
     * @param l1RegularizationWeight    The l2 regularization weight
     * @param l2RegularizationWeight    The L1 regularization weight
     */
    static inline void sortLabelWiseCriteria(
      SparseArrayVector<float64>::iterator tmpIterator,
      DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator,
      DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessianIterator, uint32 numLabels,
      uint32 numPredictions, float64 l1RegularizationWeight, float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < numLabels; i++) {
            IndexedValue<float64>& entry = tmpIterator[i];
            entry.index = i;
            entry.value = calculateLabelWiseScore(gradientIterator[i], hessianIterator[i], l1RegularizationWeight,
                                                  l2RegularizationWeight);
        }

        std::partial_sort(tmpIterator, &tmpIterator[numPredictions], &tmpIterator[numLabels],
                          CompareLabelWiseCriteria());
    }

}