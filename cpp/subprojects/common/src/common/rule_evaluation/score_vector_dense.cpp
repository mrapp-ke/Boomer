#include "common/rule_evaluation/score_vector_dense.hpp"
#include "common/rule_evaluation/score_processor.hpp"
#include "common/head_refinement/prediction.hpp"
#include "common/indices/index_vector_full.hpp"
#include "common/indices/index_vector_partial.hpp"


template<class T>
DenseScoreVector<T>::DenseScoreVector(const T& labelIndices)
    : labelIndices_(labelIndices), predictedScoreVector_(DenseVector<float64>(labelIndices.getNumElements())) {

}

template<class T>
typename DenseScoreVector<T>::index_const_iterator DenseScoreVector<T>::indices_cbegin() const {
    return labelIndices_.cbegin();
}

template<class T>
typename DenseScoreVector<T>::index_const_iterator DenseScoreVector<T>::indices_cend() const {
    return labelIndices_.cend();
}

template<class T>
typename DenseScoreVector<T>::score_iterator DenseScoreVector<T>::scores_begin() {
    return predictedScoreVector_.begin();
}

template<class T>
typename DenseScoreVector<T>::score_iterator DenseScoreVector<T>::scores_end() {
    return predictedScoreVector_.end();
}

template<class T>
typename DenseScoreVector<T>::score_const_iterator DenseScoreVector<T>::scores_cbegin() const {
    return predictedScoreVector_.cbegin();
}

template<class T>
typename DenseScoreVector<T>::score_const_iterator DenseScoreVector<T>::scores_cend() const {
    return predictedScoreVector_.cend();
}

template<class T>
uint32 DenseScoreVector<T>::getNumElements() const {
    return predictedScoreVector_.getNumElements();
}

template<class T>
bool DenseScoreVector<T>::isPartial() const {
    return labelIndices_.isPartial();
}

template<class T>
void DenseScoreVector<T>::updatePrediction(AbstractPrediction& prediction) const {
    prediction.set(predictedScoreVector_.cbegin(), predictedScoreVector_.cend());
}

template<class T>
const AbstractEvaluatedPrediction* DenseScoreVector<T>::processScores(const AbstractEvaluatedPrediction* bestHead,
                                                                      IScoreProcessor& scoreProcessor) const {
    return scoreProcessor.processScores(bestHead, *this);
}

template class DenseScoreVector<PartialIndexVector>;
template class DenseScoreVector<FullIndexVector>;
