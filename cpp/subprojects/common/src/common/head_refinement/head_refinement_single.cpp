#include "common/head_refinement/head_refinement_single.hpp"
#include "common/head_refinement/prediction_partial.hpp"
#include "common/rule_evaluation/score_processor_label_wise.hpp"


/**
 * Allows to find the best single-label head that predicts for a single label.
 */
class SingleLabelHeadRefinement final : public IHeadRefinement, public ILabelWiseScoreProcessor {

    private:

        std::unique_ptr<PartialPrediction> headPtr_;

        template<class T>
        const AbstractEvaluatedPrediction* processScoresInternally(const AbstractEvaluatedPrediction* bestHead,
                                                                   const T& scoreVector) {
            uint32 numPredictions = scoreVector.getNumElements();
            typename T::quality_score_const_iterator qualityScoreIterator = scoreVector.quality_scores_cbegin();
            uint32 bestC = 0;
            float64 bestQualityScore = qualityScoreIterator[bestC];

            for (uint32 c = 1; c < numPredictions; c++) {
                float64 qualityScore = qualityScoreIterator[c];

                if (qualityScore < bestQualityScore) {
                    bestQualityScore = qualityScore;
                    bestC = c;
                }
            }

            // The quality score must be better than that of `bestHead`...
            if (bestHead == nullptr || bestQualityScore < bestHead->overallQualityScore) {
                typename T::score_const_iterator scoreIterator = scoreVector.scores_cbegin();
                typename T::index_const_iterator indexIterator = scoreVector.indices_cbegin();

                if (headPtr_.get() == nullptr) {
                    headPtr_ = std::make_unique<PartialPrediction>(1);
                }

                PartialPrediction::score_iterator headScoreIterator = headPtr_->scores_begin();
                PartialPrediction::index_iterator headIndexIterator = headPtr_->indices_begin();
                headScoreIterator[0] = scoreIterator[bestC];
                headIndexIterator[0] = indexIterator[bestC];
                headPtr_->overallQualityScore = bestQualityScore;
                return headPtr_.get();
            }

            return nullptr;
        }

    public:

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

        const AbstractEvaluatedPrediction* processScores(
                const AbstractEvaluatedPrediction* bestHead,
                const DenseBinnedLabelWiseScoreVector<FullIndexVector>& scoreVector) override {
            return processScoresInternally<DenseBinnedLabelWiseScoreVector<FullIndexVector>>(bestHead, scoreVector);
        }

        const AbstractEvaluatedPrediction* processScores(
                const AbstractEvaluatedPrediction* bestHead,
                const DenseBinnedLabelWiseScoreVector<PartialIndexVector>& scoreVector) override {
            return processScoresInternally<DenseBinnedLabelWiseScoreVector<PartialIndexVector>>(bestHead, scoreVector);
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

std::unique_ptr<IHeadRefinement> SingleLabelHeadRefinementFactory::create(const FullIndexVector& labelIndices) const {
    return std::make_unique<SingleLabelHeadRefinement>();
}

std::unique_ptr<IHeadRefinement> SingleLabelHeadRefinementFactory::create(
        const PartialIndexVector& labelIndices) const {
    return std::make_unique<SingleLabelHeadRefinement>();
}
