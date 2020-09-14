#include "example_wise_losses.h"
#include <math.h>

using namespace boosting;


AbstractExampleWiseLoss::~AbstractExampleWiseLoss() {

}

void AbstractExampleWiseLoss::calculateGradientsAndHessians(AbstractRandomAccessLabelMatrix* labelMatrix,
                                                            uint32 exampleIndex, const float64* predictedScores,
                                                            float64* gradients, float64* hessians) {

}

ExampleWiseLogisticLossImpl::~ExampleWiseLogisticLossImpl() {

}

void ExampleWiseLogisticLossImpl::calculateGradientsAndHessians(AbstractRandomAccessLabelMatrix* labelMatrix,
                                                                uint32 exampleIndex, const float64* predictedScores,
                                                                float64* gradients, float64* hessians) {
    uint32 numLabels = labelMatrix->numLabels_;
    float64 sumOfExponentials = 1;

    for (uint32 c = 0; c < numLabels; c++) {
        uint8 trueLabel = labelMatrix->getLabel(exampleIndex, c);
        float64 expectedScore = trueLabel ? 1 : -1;
        float64 predictedScore = predictedScores[c];
        float64 exponential = exp(-expectedScore * predictedScore);
        gradients[c] = exponential;  // Temporarily store the exponential in the existing output array
        sumOfExponentials += exponential;
    }

    float64 sumOfExponentialsPow = pow(sumOfExponentials, 2);
    uint32 i = 0;

    for (uint32 c = 0; c < numLabels; c++) {
        uint8 trueLabel = labelMatrix->getLabel(exampleIndex, c);
        float64 expectedScore = trueLabel ? 1 : -1;
        float64 predictedScore = predictedScores[c];
        float64 exponential = gradients[c];
        float64 tmp = (-expectedScore * exponential) / sumOfExponentials;
        gradients[c] = tmp;

        for (uint32 c2 = 0; c2 < c; c2++) {
            trueLabel = labelMatrix->getLabel(exampleIndex, c2);
            float64 expectedScore2 = trueLabel ? 1 : -1;
            float64 predictedScore2 = predictedScores[c2];
            tmp = exp((-expectedScore2 * predictedScore2) - (expectedScore * predictedScore));
            tmp *= -expectedScore2 * expectedScore;
            tmp /= sumOfExponentialsPow;
            hessians[i] = tmp;
            i++;
        }

        tmp = pow(expectedScore, 2) * exponential * (sumOfExponentials - exponential);
        tmp /= sumOfExponentialsPow;
        hessians[i] = tmp;
        i++;
    }
}
