#include "seco/head_refinement/head_refinement_partial.hpp"
#include "common/data/vector_sparse_array.hpp"
#include "common/head_refinement/prediction_partial.hpp"
#include "common/rule_evaluation/score_processor_label_wise.hpp"
#include <algorithm>


namespace seco {

    template<class Iterator>
    static inline std::unique_ptr<SparseArrayVector<float64>> argsort(Iterator iterator, uint32 numElements) {
        std::unique_ptr<SparseArrayVector<float64>> sortedVectorPtr = std::make_unique<SparseArrayVector<float64>>(
            numElements);
        SparseArrayVector<float64>::iterator sortedIterator = sortedVectorPtr->begin();

        for (uint32 i = 0; i < numElements; i++) {
            sortedIterator[i].index = i;
            sortedIterator[i].value = iterator[i];
        }

        sortedVectorPtr->sortByValues();
        return sortedVectorPtr;
    }

    /**
     * Allows to find the best head that predicts for one or several labels depending on a lift function.
     *
     * @tparam T The type of the vector that provides access to the indices of the labels that are considered when
     *           searching for the best head
     */
    template<class T>
    class PartialHeadRefinement final : public IHeadRefinement, public ILabelWiseScoreProcessor {

        private:

            bool keepLabels_;

            std::shared_ptr<ILiftFunction> liftFunctionPtr_;

            std::unique_ptr<PartialPrediction> headPtr_;

            template<class T2>
            const AbstractEvaluatedPrediction* processScoresInternally(const AbstractEvaluatedPrediction* bestHead,
                                                                       const T2& scoreVector) {
                uint32 numPredictions = scoreVector.getNumElements();
                typename T2::quality_score_const_iterator qualityScoreIterator = scoreVector.quality_scores_cbegin();
                std::unique_ptr<SparseArrayVector<float64>> sortedVectorPtr;
                float64 sumOfQualityScores = 0;
                uint32 bestNumPredictions = 0;
                float64 bestQualityScore = 0;

                if (keepLabels_) {
                    for (uint32 c = 0; c < numPredictions; c++) {
                        sumOfQualityScores += 1 - qualityScoreIterator[c];
                    }

                    bestQualityScore =
                        1 - (sumOfQualityScores / numPredictions) * liftFunctionPtr_->calculateLift(numPredictions);
                    bestNumPredictions = numPredictions;
                } else {
                    sortedVectorPtr = argsort(qualityScoreIterator, numPredictions);
                    SparseArrayVector<float64>::const_iterator sortedIterator = sortedVectorPtr->cbegin();
                    float64 maximumLift = liftFunctionPtr_->getMaxLift();

                    for (uint32 c = 0; c < numPredictions; c++) {
                        sumOfQualityScores += 1 - qualityScoreIterator[sortedIterator[c].index];
                        float64 qualityScore = 1 - (sumOfQualityScores / (c + 1))
                                               * liftFunctionPtr_->calculateLift(c + 1);

                        if (c == 0 || qualityScore < bestQualityScore) {
                            bestNumPredictions = c + 1;
                            bestQualityScore = qualityScore;
                        }

                        if (qualityScore * maximumLift < bestQualityScore) {
                            // Prunable by decomposition...
                            break;
                        }
                    }
                }

                if (bestHead == nullptr || bestQualityScore < bestHead->overallQualityScore) {
                    if (headPtr_.get() == nullptr) {
                        headPtr_ = std::make_unique<PartialPrediction>(bestNumPredictions);
                    } else if (headPtr_->getNumElements() != bestNumPredictions) {
                        headPtr_->setNumElements(bestNumPredictions, false);
                    }

                    if (keepLabels_) {
                        std::copy(scoreVector.indices_cbegin(), scoreVector.indices_cend(), headPtr_->indices_begin());
                        std::copy(scoreVector.scores_cbegin(), scoreVector.scores_cend(), headPtr_->scores_begin());
                    } else {
                        SparseArrayVector<float64>::const_iterator sortedIterator = sortedVectorPtr->cbegin();
                        typename T2::score_const_iterator scoreIterator = scoreVector.scores_cbegin();
                        typename T2::index_const_iterator indexIterator = scoreVector.indices_cbegin();
                        PartialPrediction::score_iterator headScoreIterator = headPtr_->scores_begin();
                        PartialPrediction::index_iterator headIndexIterator = headPtr_->indices_begin();

                        for (uint32 c = 0; c < bestNumPredictions; c++) {
                            uint32 i = sortedIterator[c].index;
                            headIndexIterator[c] = indexIterator[i];
                            headScoreIterator[c] = scoreIterator[i];
                        }
                    }

                    headPtr_->overallQualityScore = bestQualityScore;
                    return headPtr_.get();
                }

                return nullptr;
            }

        public:

            /**
             * @param labelIndices      A reference to an object of template type `T` that provides access to the
             *                          indices of the labels that should be considered when searching for the best head
             * @param liftFunctionPtr   A shared pointer to an object of type `ILiftFunction` that should affect the
             *                          quality scores of rules, depending on how many labels they predict
             */
            PartialHeadRefinement(const T& labelIndices, std::shared_ptr<ILiftFunction> liftFunctionPtr)
                : keepLabels_(labelIndices.isPartial()), liftFunctionPtr_(liftFunctionPtr) {

            }

            const AbstractEvaluatedPrediction* processScores(
                    const AbstractEvaluatedPrediction* bestHead,
                    const DenseLabelWiseScoreVector<FullIndexVector>& scoreVector) override {
                return processScoresInternally<DenseLabelWiseScoreVector<FullIndexVector>>(bestHead, scoreVector);
            }

            const AbstractEvaluatedPrediction* processScores(
                    const AbstractEvaluatedPrediction* bestHead,
                    const DenseLabelWiseScoreVector<PartialIndexVector>& scoreVector) override {
                return processScoresInternally<DenseLabelWiseScoreVector<PartialIndexVector>>(bestHead, scoreVector);
            }

            const AbstractEvaluatedPrediction* findHead(const AbstractEvaluatedPrediction* bestHead,
                                                        IStatisticsSubset& statisticsSubset, bool uncovered,
                                                        bool accumulated) override {
                const ILabelWiseScoreVector& scoreVector = statisticsSubset.calculateLabelWisePrediction(uncovered,
                                                                                                         accumulated);
                return scoreVector.processScores(bestHead, *this);
            }

            std::unique_ptr<AbstractEvaluatedPrediction> pollHead() override {
                return std::move(headPtr_);
            }

            const IScoreVector& calculatePrediction(IStatisticsSubset& statisticsSubset, bool uncovered,
                                                    bool accumulated) const override {
                return statisticsSubset.calculateLabelWisePrediction(uncovered, accumulated);
            }

    };

    PartialHeadRefinementFactory::PartialHeadRefinementFactory(std::shared_ptr<ILiftFunction> liftFunctionPtr)
        : liftFunctionPtr_(liftFunctionPtr) {

    }

    std::unique_ptr<IHeadRefinement> PartialHeadRefinementFactory::create(const FullIndexVector& labelIndices) const {
        return std::make_unique<PartialHeadRefinement<FullIndexVector>>(labelIndices, liftFunctionPtr_);
    }

    std::unique_ptr<IHeadRefinement> PartialHeadRefinementFactory::create(
            const PartialIndexVector& labelIndices) const {
        return std::make_unique<PartialHeadRefinement<PartialIndexVector>>(labelIndices, liftFunctionPtr_);
    }

}
