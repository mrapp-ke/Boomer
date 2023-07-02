/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/data/statistic_vector_label_wise_dense.hpp"
#include "common/data/vector_sparse_array.hpp"
#include "rule_evaluation_label_wise_common.hpp"

#include <algorithm>

namespace boosting {

    /**
     * Allows to compare two objects of type `IndexedValue` that store the optimal prediction for a labels, as well as
     * its index, according to the following strict weak ordering: If the absolute value of the first object is greater,
     * it goes before the second one.
     */
    struct CompareLabelWiseCriteria final {
        public:

            /**
             * Returns whether the a given object of type `IndexedValue` that stores the optimal prediction for a label,
             * as well as its index, should go before a second one.
             *
             * @param lhs   A reference to a first object of type `IndexedValue`
             * @param rhs   A reference to a second object of type `IndexedValue`
             * @return      True, if the first object should go before the second one, false otherwise
             */
            inline bool operator()(const IndexedValue<float64>& lhs, const IndexedValue<float64>& rhs) const {
                return std::abs(lhs.value) > std::abs(rhs.value);
            }
    };

    /**
     * Calculates the scores to be predicted for individual labels and sorts them by their quality, such that the first
     * `numPredictions` elements are the best-rated ones.
     *
     * @tparam StatisticIterator        The type of the iterator that provides access to the gradients and Hessians
     * @param tmpIterator               An iterator that provides random access to a temporary array, which should be
     *                                  used to store the sorted scores and their original indices
     * @param statisticIterator         An iterator that provides access to the gradients and Hessians for each label
     * @param numLabels                 The total number of available labels
     * @param numPrediction             The number of the best-rated predictions to be determined
     * @param l1RegularizationWeight    The l2 regularization weight
     * @param l2RegularizationWeight    The L1 regularization weight
     */
    template<typename StatisticIterator>
    static inline void sortLabelWiseScores(SparseArrayVector<float64>::iterator tmpIterator,
                                           StatisticIterator& statisticIterator, uint32 numLabels,
                                           uint32 numPredictions, float64 l1RegularizationWeight,
                                           float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < numLabels; i++) {
            const Tuple<float64>& tuple = statisticIterator[i];
            IndexedValue<float64>& entry = tmpIterator[i];
            entry.index = i;
            entry.value =
              calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight, l2RegularizationWeight);
        }

        std::partial_sort(tmpIterator, &tmpIterator[numPredictions], &tmpIterator[numLabels],
                          CompareLabelWiseCriteria());
    }

}