/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/prediction/discretization_function.hpp"
#include "boosting/prediction/transformation_binary.hpp"

namespace boosting {

    /**
     * An implementation of the class `IBinaryTransformation` that transforms regression scores that are predicted for
     * individual labels into binary predictions via element-wise application of an `IDiscretizationFunction`.
     */
    class LabelWiseBinaryTransformation final : public IBinaryTransformation {
        private:

            std::unique_ptr<IDiscretizationFunction> discretizationFunctionPtr_;

        public:

            /**
             * @param discretizationFunctionPtr An unique pointer to an object of type `IDiscretizationFunction` that
             *                                  should be used to discretize regression scores
             */
            LabelWiseBinaryTransformation(std::unique_ptr<IDiscretizationFunction> discretizationFunctionPtr);

            void apply(VectorConstView<float64>::const_iterator scoresBegin,
                       VectorConstView<float64>::const_iterator scoresEnd, VectorView<uint8>::iterator predictionBegin,
                       VectorView<uint8>::iterator predictionEnd) const override;

            void apply(VectorConstView<float64>::const_iterator scoresBegin,
                       VectorConstView<float64>::const_iterator scoresEnd,
                       BinaryLilMatrix::row predictionRow) const override;
    };

}
