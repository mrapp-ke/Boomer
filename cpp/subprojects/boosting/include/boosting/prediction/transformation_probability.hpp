/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_vector.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to transform aggregated scores into probability estimates.
     */
    class IProbabilityTransformation {
        public:

            virtual ~IProbabilityTransformation() {};

            /**
             * Transforms aggregated scores into probability estimates.
             *
             * @param scoresBegin           An iterator of type `VectorConstView::const_iterator` to the beginning of
             *                              the aggregated scores
             * @param scoresEnd             An iterator of type `VectorConstView::const_iterator` to the end of the
             *                              the aggregated scores
             * @param probabilitiesBegin    An iterator of type `VectorView::iterator` to the beginning of the
             *                              probabilities
             * @param probabilitiesEnd      An iterator of type `VectorView::iterator` to the end of the probabilities
             */
            virtual void apply(VectorConstView<float64>::const_iterator scoresBegin,
                               VectorConstView<float64>::const_iterator scoresEnd,
                               VectorView<float64>::iterator probabilitiesBegin,
                               VectorView<float64>::iterator probabilitiesEnd) const = 0;
    };

}
