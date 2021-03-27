/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/losses/loss_label_wise.hpp"


namespace boosting {

    /**
     * A multi-label variant of the logistic loss that is applied label-wise.
     */
    class LabelWiseLogisticLoss final : public AbstractLabelWiseLoss {

        protected:

            void updateGradientAndHessian(DenseVector<float64>::iterator gradient,
                                          DenseVector<float64>::iterator hessian, bool trueLabel,
                                          float64 predictedScore) const override;

            float64 evaluate(bool trueLabel, float64 predictedScore) const override;

    };

}
