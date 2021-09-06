/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/math/math.hpp"


namespace boosting {

    /**
     * Calculates and returns the optimal score to be predicted for a single label, based on the corresponding gradient
     * and Hessian and taking L2 regularization into account.
     *
     * @param gradient                  The gradient that corresponds to the label
     * @param hessian                   The Hessian that corresponds to the label
     * @param l2RegularizationWeight    The weight of the L2 regularization
     * @return                          The predicted score that has been calculated
     */
    static inline constexpr float64 calculateLabelWiseScore(float64 gradient, float64 hessian,
                                                            float64 l2RegularizationWeight) {
        return divideOrZero(-gradient, hessian + l2RegularizationWeight);
    }

    /**
     * Calculates and returns a quality score that assesses the quality of the score that is predicted for a single
     * label.
     *
     * @param score                     The predicted score
     * @param gradient                  The gradient
     * @param hessian                   The Hessian
     * @param l2RegularizationWeight    The weight of the L2 regularization
     * @return                          The quality score that has been calculated
     */
    static inline constexpr float64 calculateLabelWiseQualityScore(float64 score, float64 gradient, float64 hessian,
                                                                   float64 l2RegularizationWeight) {
        float64 scorePow = score * score;
        return (gradient * score) + (0.5 * hessian * scorePow) + (0.5 * l2RegularizationWeight * scorePow);
    }


}
