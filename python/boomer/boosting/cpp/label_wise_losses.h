/**
 * Implements different differentiable loss functions that are applied label-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/input_data.h"
#include <utility>


namespace boosting {

    /**
     * An abstract base class for all (decomposable) loss functions that are applied label-wise.
     */
    class AbstractLabelWiseLoss {

        public:

            virtual ~AbstractLabelWiseLoss();

            /**
             * Must be implemented by subclasses to calculate the gradient (first derivative) and Hessian (second
             * derivative) of the loss function for a certain example and label.
             *
             * @param labelMatrix       A pointer to an object of type `AbstractRandomAccessLabelMatrix` that provides
             *                          random access to the labels of the training examples
             * @param exampleIndex      The index of the example for which the gradient and Hessian should be calculated
             * @param labelIndex        The index of the label for which the gradient and Hessian should be calculated
             * @param predictedScore    A scalar of type `float64`, representing the score that is predicted for the
             *                          respective example and label
             * @return                  A pair that contains two scalars of type `float64`, representing the gradient
             *                          and the Hessian that have been calculated
             */
            virtual std::pair<float64, float64> calculateGradientAndHessian(
                    AbstractRandomAccessLabelMatrix* labelMatrix, uint32 exampleIndex, uint32 labelIndex,
                    float64 predictedScore);

    };

    /**
     * A multi-label variant of the logistic loss that is applied label-wise.
     */
    class LabelWiseLogisticLossImpl : public AbstractLabelWiseLoss {

        public:

            ~LabelWiseLogisticLossImpl();

            std::pair<float64, float64> calculateGradientAndHessian(AbstractRandomAccessLabelMatrix* labelMatrix,
                                                                    uint32 exampleIndex, uint32 labelIndex,
                                                                    float64 predictedScore) override;

    };

    /**
     * A multi-label variant of the squared error loss that is applied label-wise.
     */
    class LabelWiseSquaredErrorLossImpl : public AbstractLabelWiseLoss {

        public:

            ~LabelWiseSquaredErrorLossImpl();

            std::pair<float64, float64> calculateGradientAndHessian(AbstractRandomAccessLabelMatrix* labelMatrix,
                                                                    uint32 exampleIndex, uint32 labelIndex,
                                                                    float64 predictedScore) override;

    };

}
