#include "boosting/prediction/transformation_binary_example_wise.hpp"

#include "common/iterator/binary_forward_iterator.hpp"

namespace boosting {

    ExampleWiseBinaryTransformation::ExampleWiseBinaryTransformation(
      const LabelVectorSet& labelVectorSet, std::unique_ptr<IDistanceMeasure> distanceMeasurePtr)
        : labelVectorSet_(labelVectorSet), distanceMeasurePtr_(std::move(distanceMeasurePtr)) {}

    void ExampleWiseBinaryTransformation::apply(VectorConstView<float64>::const_iterator scoresBegin,
                                                VectorConstView<float64>::const_iterator scoresEnd,
                                                VectorView<uint8>::iterator predictionBegin,
                                                VectorView<uint8>::iterator predictionEnd) const {
        const LabelVector& labelVector =
          distanceMeasurePtr_->getClosestLabelVector(labelVectorSet_, scoresBegin, scoresEnd);
        uint32 numLabels = predictionEnd - predictionBegin;
        auto labelIterator = make_binary_forward_iterator(labelVector.cbegin(), labelVector.cend());

        for (uint32 i = 0; i < numLabels; i++) {
            bool label = *labelIterator;
            predictionBegin[i] = label ? 1 : 0;
            labelIterator++;
        }
    }

    void ExampleWiseBinaryTransformation::apply(VectorConstView<float64>::const_iterator scoresBegin,
                                                VectorConstView<float64>::const_iterator scoresEnd,
                                                BinaryLilMatrix::row predictionRow) const {
        const LabelVector& labelVector =
          distanceMeasurePtr_->getClosestLabelVector(labelVectorSet_, scoresBegin, scoresEnd);
        uint32 numIndices = labelVector.getNumElements();
        LabelVector::const_iterator indexIterator = labelVector.cbegin();
        predictionRow.reserve(numIndices);

        for (uint32 i = 0; i < numIndices; i++) {
            uint32 labelIndex = indexIterator[i];
            predictionRow.emplace_back(labelIndex);
        }
    }

}
