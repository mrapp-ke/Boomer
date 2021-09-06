#include "boosting/losses/loss_label_wise_squared_error.hpp"


namespace boosting {

    static inline void updateGradientAndHessian(bool trueLabel, float64 predictedScore, float64* gradient,
                                                float64* hessian) {
        float64 expectedScore = trueLabel ? 1 : -1;
        *gradient = (2 * predictedScore) - (2 * expectedScore);
        *hessian = 2;
    }

    static inline float64 evaluatePrediction(bool trueLabel, float64 predictedScore) {
        float64 expectedScore = trueLabel ? 1 : -1;
        float64 difference = (expectedScore - predictedScore);
        return difference * difference;
    }

    LabelWiseSquaredErrorLoss::LabelWiseSquaredErrorLoss()
        : AbstractLabelWiseLoss(&updateGradientAndHessian, &evaluatePrediction) {

    }

}
