#include "seco/rule_evaluation/lift_function_peak.hpp"
#include "common/validation.hpp"
#include <cmath>


namespace seco {

    PeakLiftFunction::PeakLiftFunction(uint32 numLabels, uint32 peakLabel, float64 maxLift, float64 curvature)
        : numLabels_(numLabels), peakLabel_(peakLabel), maxLift_(maxLift), exponent_(1.0 / curvature) {
        assertGreater<uint32>("numLabels", numLabels, 0);
        assertGreaterOrEqual<uint32>("peakLabel", peakLabel, 0);
        assertLessOrEqual<uint32>("peakLabel", peakLabel, numLabels);
        assertGreaterOrEqual<float64>("maxLift", maxLift, 1);
        assertGreater<float64>("curvature", curvature, 0);
    }

    float64 PeakLiftFunction::calculateLift(uint32 numLabels) const {
        float64 normalization;

        if (numLabels < peakLabel_) {
            normalization = ((float64) numLabels - 1) / ((float64) peakLabel_ - 1);
        } else if (numLabels > peakLabel_) {
            normalization = ((float64) numLabels - (float64) numLabels_)
                            / ((float64) numLabels_ - (float64) peakLabel_);
        } else {
            return maxLift_;
        }

        return 1 + pow(normalization, exponent_) * (maxLift_ - 1);
    }

    float64 PeakLiftFunction::getMaxLift() const {
        return maxLift_;
    }

}
