/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/transformation_binary.hpp"
#include "common/measures/measure_distance.hpp"
#include "common/prediction/label_vector_set.hpp"

namespace boosting {

    /**
     * An implementation of the class `IBinaryTransformation` that transforms regression scores into binary predictions
     * by comparing the scores to the known label vectors according to a certain distance measure and picking the
     * closest one.
     */
    class ExampleWiseBinaryTransformation final : public IBinaryTransformation {
        private:

            const LabelVectorSet& labelVectorSet_;

            const std::unique_ptr<IDistanceMeasure> distanceMeasurePtr_;

        public:

            /**
             * @param labelVectorSet        A reference to an object of type `LabelVectorSet` that stores all known
             *                              label vectors
             * @param distanceMeasurePtr    An unique pointer to an object of type `IDistanceMeasure` that implements
             *                              the distance measure for comparing regression scores to known label vectors
             */
            ExampleWiseBinaryTransformation(const LabelVectorSet& labelVectorSet,
                                            std::unique_ptr<IDistanceMeasure> distanceMeasurePtr);

            void apply(VectorConstView<float64>::const_iterator scoresBegin,
                       VectorConstView<float64>::const_iterator scoresEnd, VectorView<uint8>::iterator predictionBegin,
                       VectorView<uint8>::iterator predictionEnd) const override;

            void apply(VectorConstView<float64>::const_iterator scoresBegin,
                       VectorConstView<float64>::const_iterator scoresEnd,
                       BinaryLilMatrix::row predictionRow) const override;
    };

}
