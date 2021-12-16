#include "common/rule_refinement/prediction.hpp"
#include "common/data/arrays.hpp"


AbstractPrediction::AbstractPrediction(uint32 numElements)
    : predictedScoreVector_(DenseVector<float64>(numElements)) {

}

uint32 AbstractPrediction::getNumElements() const {
    return predictedScoreVector_.getNumElements();
}

void AbstractPrediction::setNumElements(uint32 numElements, bool freeMemory) {
    predictedScoreVector_.setNumElements(numElements, freeMemory);
}

AbstractPrediction::score_iterator AbstractPrediction::scores_begin() {
    return predictedScoreVector_.begin();
}

AbstractPrediction::score_iterator AbstractPrediction::scores_end() {
    return predictedScoreVector_.end();
}

AbstractPrediction::score_const_iterator AbstractPrediction::scores_cbegin() const {
    return predictedScoreVector_.cbegin();
}

AbstractPrediction::score_const_iterator AbstractPrediction::scores_cend() const {
    return predictedScoreVector_.cend();
}

void AbstractPrediction::set(AbstractPrediction::score_const_iterator begin,
                             AbstractPrediction::score_const_iterator end) {
    copyArray(begin, predictedScoreVector_.begin(), predictedScoreVector_.getNumElements());
}

void AbstractPrediction::set(DenseBinnedVector<float64>::const_iterator begin,
                             DenseBinnedVector<float64>::const_iterator end) {
    copyArray(begin, predictedScoreVector_.begin(), predictedScoreVector_.getNumElements());
}
