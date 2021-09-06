#include "common/rule_evaluation/score_vector_dense.hpp"
#include "common/rule_refinement/prediction.hpp"
#include "common/rule_refinement/score_processor.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"


template<typename T>
DenseScoreVector<T>::DenseScoreVector(const T& labelIndices)
    : labelIndices_(labelIndices), predictedScoreVector_(DenseVector<float64>(labelIndices.getNumElements())) {

}

template<typename T>
typename DenseScoreVector<T>::index_const_iterator DenseScoreVector<T>::indices_cbegin() const {
    return labelIndices_.cbegin();
}

template<typename T>
typename DenseScoreVector<T>::index_const_iterator DenseScoreVector<T>::indices_cend() const {
    return labelIndices_.cend();
}

template<typename T>
typename DenseScoreVector<T>::score_iterator DenseScoreVector<T>::scores_begin() {
    return predictedScoreVector_.begin();
}

template<typename T>
typename DenseScoreVector<T>::score_iterator DenseScoreVector<T>::scores_end() {
    return &predictedScoreVector_.begin()[labelIndices_.getNumElements()];
}

template<typename T>
typename DenseScoreVector<T>::score_const_iterator DenseScoreVector<T>::scores_cbegin() const {
    return predictedScoreVector_.cbegin();
}

template<typename T>
typename DenseScoreVector<T>::score_const_iterator DenseScoreVector<T>::scores_cend() const {
    return &predictedScoreVector_.cbegin()[labelIndices_.getNumElements()];
}

template<typename T>
uint32 DenseScoreVector<T>::getNumElements() const {
    return labelIndices_.getNumElements();
}

template<typename T>
bool DenseScoreVector<T>::isPartial() const {
    return labelIndices_.isPartial();
}

template<typename T>
void DenseScoreVector<T>::updatePrediction(AbstractPrediction& prediction) const {
    prediction.set(this->scores_cbegin(), this->scores_cend());
}

template<typename T>
const AbstractEvaluatedPrediction* DenseScoreVector<T>::processScores(ScoreProcessor& scoreProcessor) const {
    return scoreProcessor.processScores(*this);
}

template class DenseScoreVector<PartialIndexVector>;
template class DenseScoreVector<CompleteIndexVector>;
