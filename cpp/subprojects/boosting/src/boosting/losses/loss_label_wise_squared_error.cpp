#include "boosting/losses/loss_label_wise_squared_error.hpp"


namespace boosting {

    void LabelWiseSquaredErrorLoss::updateGradientAndHessian(DenseVector<float64>::iterator gradient,
                                                             DenseVector<float64>::iterator hessian, bool trueLabel,
                                                             float64 predictedScore) const {
        float64 expectedScore = trueLabel ? 1 : -1;
        *gradient = (2 * predictedScore) - (2 * expectedScore);
        *hessian = 2;
    }

    float64 LabelWiseSquaredErrorLoss::evaluate(bool trueLabel, float64 predictedScore) const {
        float64 expectedScore = trueLabel ? 1 : -1;
        float64 difference = (expectedScore - predictedScore);
        return difference * difference;
    }

}
