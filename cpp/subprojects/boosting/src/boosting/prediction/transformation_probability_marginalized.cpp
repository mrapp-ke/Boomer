#include "boosting/prediction/transformation_probability_marginalized.hpp"

#include "common/data/arrays.hpp"

namespace boosting {

    MarginalizedProbabilityTransformation::MarginalizedProbabilityTransformation(
      const LabelVectorSet& labelVectorSet, std::unique_ptr<IJointProbabilityFunction> jointProbabilityFunctionPtr)
        : labelVectorSet_(labelVectorSet), jointProbabilityFunctionPtr_(std::move(jointProbabilityFunctionPtr)) {}

    void MarginalizedProbabilityTransformation::apply(VectorConstView<float64>::const_iterator scoresBegin,
                                                      VectorConstView<float64>::const_iterator scoresEnd,
                                                      VectorView<float64>::iterator probabilitiesBegin,
                                                      VectorView<float64>::iterator probabilitiesEnd) const {
        std::unique_ptr<DenseVector<float64>> jointProbabilityVectorPtr =
          jointProbabilityFunctionPtr_->transformScoresIntoJointProbabilities(labelVectorSet_, scoresBegin, scoresEnd);
        DenseVector<float64>::const_iterator jointProbabilityIterator = jointProbabilityVectorPtr->cbegin();
        uint32 numLabels = probabilitiesEnd - probabilitiesBegin;
        setArrayToZeros(probabilitiesBegin, numLabels);
        LabelVectorSet::const_iterator labelVectorIterator = labelVectorSet_.cbegin();
        uint32 numLabelVectors = labelVectorSet_.getNumLabelVectors();

        for (uint32 i = 0; i < numLabelVectors; i++) {
            const LabelVector& labelVector = *labelVectorIterator[i];
            uint32 numRelevantLabels = labelVector.getNumElements();
            LabelVector::const_iterator labelIndexIterator = labelVector.cbegin();
            float64 jointProbability = jointProbabilityIterator[i];

            for (uint32 j = 0; j < numRelevantLabels; j++) {
                uint32 labelIndex = labelIndexIterator[j];
                probabilitiesBegin[labelIndex] += jointProbability;
            }
        }
    }

}
