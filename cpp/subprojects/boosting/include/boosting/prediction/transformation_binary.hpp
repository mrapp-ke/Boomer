/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/matrix_lil_binary.hpp"
#include "common/data/view_vector.hpp"

namespace boosting {

    /**
     * Defines an interface for all classes that allow to transform regression scores into binary predictions.
     */
    class IBinaryTransformation {
        public:

            virtual ~IBinaryTransformation() {};

            /**
             * Transforms regression scores into binary predictions.
             *
             * @param scoresBegin       An iterator of type `VectorConstView::const_iterator` to the beginning of the
             *                          regression scores
             * @param scoresEnd         An iterator of type `VectorConstView::const_iterator` to the end of the
             *                          regression scores
             * @param predictionBegin   An iterator of type `VectorView::iterator` to the beginning of the binary
             *                          predictions
             * @param predictionEnd     An iterator of type `VectorView::iterator` to the end of the binary predictions
             */
            virtual void apply(VectorConstView<float64>::const_iterator scoresBegin,
                               VectorConstView<float64>::const_iterator scoresEnd,
                               VectorView<uint8>::iterator predictionBegin,
                               VectorView<uint8>::iterator predictionEnd) const = 0;

            /**
             * Transforms regression scores into sparse binary predictions.
             *
             * @param scoresBegin   An iterator of type `VectorConstView::const_iterator` to the beginning of the
             *                      regression scores
             * @param scoresEnd     An iterator of type `VectorConstView::const_iterator` to the end of the regression
             *                      scores
             * @param predictionRow An object of type `BinaryLilMatrix::row` that should be used to store the binary
             *                      predictions
             */
            virtual void apply(VectorConstView<float64>::const_iterator scoresBegin,
                               VectorConstView<float64>::const_iterator scoresEnd,
                               BinaryLilMatrix::row predictionRow) const = 0;
    };

}
