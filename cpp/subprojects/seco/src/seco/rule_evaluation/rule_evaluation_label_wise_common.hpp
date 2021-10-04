/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "seco/data/confusion_matrix.hpp"
#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Calculates and returns a quality score that assesses the quality of the score that is predicted by a rule for a
     * single label.
     *
     * @param totalConfusionMatrix      A reference to an object of type `ConfusionMatrix` that takes into account all
     *                                  examples
     * @param coveredConfusionMatrix    A reference to an object of type `ConfusionMatrix` that takes into account all
     *                                  examples that are covered by the rule
     * @param heuristic                 The heuristic that should be used to assess the quality
     * @return                          The quality score that has been calculated
     */
    static inline float64 calculateLabelWiseQualityScore(const ConfusionMatrix& totalConfusionMatrix,
                                                         const ConfusionMatrix& coveredConfusionMatrix,
                                                         const IHeuristic& heuristic) {
        const ConfusionMatrix uncoveredConfusionMatrix = totalConfusionMatrix - coveredConfusionMatrix;
        return heuristic.evaluateConfusionMatrix(
            coveredConfusionMatrix.in, coveredConfusionMatrix.ip, coveredConfusionMatrix.rn, coveredConfusionMatrix.rp,
            uncoveredConfusionMatrix.in, uncoveredConfusionMatrix.ip, uncoveredConfusionMatrix.rn,
            uncoveredConfusionMatrix.rp);
    }

}
