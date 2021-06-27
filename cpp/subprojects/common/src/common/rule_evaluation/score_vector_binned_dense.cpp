#include "common/rule_evaluation/score_vector_binned_dense.hpp"
#include "common/rule_evaluation/score_processor.hpp"
#include "common/head_refinement/prediction.hpp"
#include "common/indices/index_vector_full.hpp"
#include "common/indices/index_vector_partial.hpp"


template<class T>
DenseBinnedScoreVector<T>::DenseBinnedScoreVector(const T& labelIndices, uint32 numBins)
    : labelIndices_(labelIndices), binnedVector_(DenseBinnedVector<float64>(labelIndices.getNumElements(), numBins)) {

}

template<class T>
typename DenseBinnedScoreVector<T>::index_const_iterator DenseBinnedScoreVector<T>::indices_cbegin() const {
    return labelIndices_.cbegin();
}

template<class T>
typename DenseBinnedScoreVector<T>::index_const_iterator DenseBinnedScoreVector<T>::indices_cend() const {
    return labelIndices_.cend();
}

template<class T>
typename DenseBinnedScoreVector<T>::score_const_iterator DenseBinnedScoreVector<T>::scores_cbegin() const {
    return binnedVector_.cbegin();
}

template<class T>
typename DenseBinnedScoreVector<T>::score_const_iterator DenseBinnedScoreVector<T>::scores_cend() const {
    return binnedVector_.cend();
}

template<class T>
typename DenseBinnedScoreVector<T>::index_binned_iterator DenseBinnedScoreVector<T>::indices_binned_begin() {
    return binnedVector_.indices_binned_begin();
}

template<class T>
typename DenseBinnedScoreVector<T>::index_binned_iterator DenseBinnedScoreVector<T>::indices_binned_end() {
    return binnedVector_.indices_binned_end();
}

template<class T>
typename DenseBinnedScoreVector<T>::index_binned_const_iterator DenseBinnedScoreVector<T>::indices_binned_cbegin() const {
    return binnedVector_.indices_binned_cbegin();
}

template<class T>
typename DenseBinnedScoreVector<T>::index_binned_const_iterator DenseBinnedScoreVector<T>::indices_binned_cend() const {
    return binnedVector_.indices_binned_cend();
}

template<class T>
typename DenseBinnedScoreVector<T>::score_binned_iterator DenseBinnedScoreVector<T>::scores_binned_begin() {
    return binnedVector_.binned_begin();
}

template<class T>
typename DenseBinnedScoreVector<T>::score_binned_iterator DenseBinnedScoreVector<T>::scores_binned_end() {
    return binnedVector_.binned_end();
}

template<class T>
typename DenseBinnedScoreVector<T>::score_binned_const_iterator DenseBinnedScoreVector<T>::scores_binned_cbegin() const {
    return binnedVector_.binned_cbegin();
}

template<class T>
typename DenseBinnedScoreVector<T>::score_binned_const_iterator DenseBinnedScoreVector<T>::scores_binned_cend() const {
    return binnedVector_.binned_cend();
}

template<class T>
uint32 DenseBinnedScoreVector<T>::getNumElements() const {
    return binnedVector_.getNumElements();
}

template<class T>
uint32 DenseBinnedScoreVector<T>::getNumBins() const {
    return binnedVector_.getNumBins();
}

template<class T>
void DenseBinnedScoreVector<T>::setNumBins(uint32 numBins, bool freeMemory) {
    binnedVector_.setNumBins(numBins, freeMemory);
}

template<class T>
bool DenseBinnedScoreVector<T>::isPartial() const {
    return labelIndices_.isPartial();
}

template<class T>
void DenseBinnedScoreVector<T>::updatePrediction(AbstractPrediction& prediction) const {
    prediction.set(binnedVector_.cbegin(), binnedVector_.cend());
}

template<class T>
const AbstractEvaluatedPrediction* DenseBinnedScoreVector<T>::processScores(const AbstractEvaluatedPrediction* bestHead,
                                                                            IScoreProcessor& scoreProcessor) const {
    return scoreProcessor.processScores(bestHead, *this);
}

template class DenseBinnedScoreVector<PartialIndexVector>;
template class DenseBinnedScoreVector<FullIndexVector>;
