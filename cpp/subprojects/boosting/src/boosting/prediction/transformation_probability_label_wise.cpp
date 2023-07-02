#include "boosting/prediction/transformation_probability_label_wise.hpp"

namespace boosting {

    LabelWiseProbabilityTransformation::LabelWiseProbabilityTransformation(
      std::unique_ptr<IMarginalProbabilityFunction> marginalProbabilityFunctionPtr)
        : marginalProbabilityFunctionPtr_(std::move(marginalProbabilityFunctionPtr)) {}

    void LabelWiseProbabilityTransformation::apply(VectorConstView<float64>::const_iterator scoresBegin,
                                                   VectorConstView<float64>::const_iterator scoresEnd,
                                                   VectorView<float64>::iterator probabilitiesBegin,
                                                   VectorView<float64>::iterator probabilitiesEnd) const {
        uint32 numScores = scoresEnd - scoresBegin;

        for (uint32 i = 0; i < numScores; i++) {
            float64 score = scoresBegin[i];
            float64 probability = marginalProbabilityFunctionPtr_->transformScoreIntoMarginalProbability(i, score);
            probabilitiesBegin[i] = probability;
        }
    }

}
