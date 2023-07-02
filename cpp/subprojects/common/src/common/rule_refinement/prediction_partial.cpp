#include "common/rule_refinement/prediction_partial.hpp"

#include "common/data/arrays.hpp"
#include "common/data/vector_sparse_array.hpp"
#include "common/model/head_partial.hpp"
#include "common/rule_refinement/rule_refinement.hpp"
#include "common/statistics/statistics.hpp"

PartialPrediction::PartialPrediction(uint32 numElements, bool sorted)
    : AbstractEvaluatedPrediction(numElements), indexVector_(PartialIndexVector(numElements)), sorted_(sorted) {}

PartialPrediction::index_iterator PartialPrediction::indices_begin() {
    return indexVector_.begin();
}

PartialPrediction::index_iterator PartialPrediction::indices_end() {
    return indexVector_.end();
}

PartialPrediction::index_const_iterator PartialPrediction::indices_cbegin() const {
    return indexVector_.cbegin();
}

PartialPrediction::index_const_iterator PartialPrediction::indices_cend() const {
    return indexVector_.cend();
}

void PartialPrediction::setNumElements(uint32 numElements, bool freeMemory) {
    this->predictedScoreVector_.setNumElements(numElements, freeMemory);
    indexVector_.setNumElements(numElements, freeMemory);
}

void PartialPrediction::setSorted(bool sorted) {
    sorted_ = sorted;
}

bool PartialPrediction::isPartial() const {
    return true;
}

uint32 PartialPrediction::getIndex(uint32 pos) const {
    return indexVector_.getIndex(pos);
}

std::unique_ptr<IStatisticsSubset> PartialPrediction::createStatisticsSubset(const IStatistics& statistics,
                                                                             const EqualWeightVector& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> PartialPrediction::createStatisticsSubset(const IStatistics& statistics,
                                                                             const BitWeightVector& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> PartialPrediction::createStatisticsSubset(
  const IStatistics& statistics, const DenseWeightVector<uint32>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> PartialPrediction::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<EqualWeightVector>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> PartialPrediction::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<BitWeightVector>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IStatisticsSubset> PartialPrediction::createStatisticsSubset(
  const IStatistics& statistics, const OutOfSampleWeightVector<DenseWeightVector<uint32>>& weights) const {
    return statistics.createSubset(indexVector_, weights);
}

std::unique_ptr<IRuleRefinement> PartialPrediction::createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                                         uint32 featureIndex) const {
    return indexVector_.createRuleRefinement(thresholdsSubset, featureIndex);
}

void PartialPrediction::apply(IStatistics& statistics, uint32 statisticIndex) const {
    statistics.applyPrediction(statisticIndex, *this);
}

void PartialPrediction::revert(IStatistics& statistics, uint32 statisticIndex) const {
    statistics.revertPrediction(statisticIndex, *this);
}

void PartialPrediction::sort() {
    if (!sorted_) {
        uint32 numElements = this->getNumElements();

        if (numElements > 1) {
            SparseArrayVector<float64> sortedVector(numElements);
            SparseArrayVector<float64>::iterator sortedIterator = sortedVector.begin();
            index_iterator indexIterator = this->indices_begin();
            score_iterator scoreIterator = this->scores_begin();

            for (uint32 i = 0; i < numElements; i++) {
                IndexedValue<float64>& entry = sortedIterator[i];
                entry.index = indexIterator[i];
                entry.value = scoreIterator[i];
            }

            std::sort(sortedIterator, sortedVector.end(), IndexedValue<float64>::CompareIndex());

            for (uint32 i = 0; i < numElements; i++) {
                const IndexedValue<float64>& entry = sortedIterator[i];
                indexIterator[i] = entry.index;
                scoreIterator[i] = entry.value;
            }
        }

        sorted_ = true;
    }
}

std::unique_ptr<IHead> PartialPrediction::createHead() const {
    uint32 numElements = this->getNumElements();
    std::unique_ptr<PartialHead> headPtr = std::make_unique<PartialHead>(numElements);
    copyArray(this->scores_cbegin(), headPtr->scores_begin(), numElements);
    copyArray(this->indices_cbegin(), headPtr->indices_begin(), numElements);
    return headPtr;
}
