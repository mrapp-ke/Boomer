/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/math/math.hpp"


namespace boosting {

    /**
     * Returns the L1 regularization weight to be added to a specific gradient.
     *
     * @param gradient                  The gradient, the L1 regularization weight should be added to
     * @param l1RegularizationWeight    The L1 regularization weight
     * @return                          The L1 regularization weight to be added to the gradient
     */
    static inline constexpr float64 getL1RegularizationWeight(float64 gradient, float64 l1RegularizationWeight) {
        if (gradient > l1RegularizationWeight) {
            return -l1RegularizationWeight;
        } else if (gradient < -l1RegularizationWeight) {
            return l1RegularizationWeight;
        } else {
            return 0;
        }
    }

    /**
     * Calculates and returns the optimal score to be predicted for a single label, based on the corresponding gradient
     * and Hessian and taking L1 and L2 regularization into account.
     *
     * @param gradient                  The gradient that corresponds to the label
     * @param hessian                   The Hessian that corresponds to the label
     * @param l1RegularizationWeight    The weight of the L1 regularization
     * @param l2RegularizationWeight    The weight of the L2 regularization
     * @return                          The predicted score that has been calculated
     */
    static inline constexpr float64 calculateLabelWiseScore(float64 gradient, float64 hessian,
                                                            float64 l1RegularizationWeight,
                                                            float64 l2RegularizationWeight) {
        return divideOrZero(-gradient + getL1RegularizationWeight(gradient, l1RegularizationWeight),
                            hessian + l2RegularizationWeight);
    }

}
