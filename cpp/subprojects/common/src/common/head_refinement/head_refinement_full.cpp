#include "common/head_refinement/head_refinement_full.hpp"
#include "common/head_refinement/prediction_full.hpp"
#include "common/head_refinement/prediction_partial.hpp"
#include "common/rule_evaluation/score_processor.hpp"
#include <algorithm>


/**
 * Allows to find the best multi-label head that predicts for all labels.
 */
class FullHeadRefinement final : public IHeadRefinement, public IScoreProcessor {

    private:

        std::unique_ptr<AbstractEvaluatedPrediction> headPtr_;

        template<class T>
        const AbstractEvaluatedPrediction* processScoresInternally(const AbstractEvaluatedPrediction* bestHead,
                                                                   const T& scoreVector) {
            float64 overallQualityScore = scoreVector.overallQualityScore;

            // The quality score must be better than that of `bestHead`...
            if (bestHead == nullptr || overallQualityScore < bestHead->overallQualityScore) {
                uint32 numPredictions = scoreVector.getNumElements();

                if (headPtr_.get() == nullptr) {
                    if (scoreVector.isPartial()) {
                        std::unique_ptr<PartialPrediction> headPtr =
                            std::make_unique<PartialPrediction>(numPredictions);
                        std::copy(scoreVector.indices_cbegin(), scoreVector.indices_cend(), headPtr->indices_begin());
                        headPtr_ = std::move(headPtr);
                    } else {
                        headPtr_ = std::make_unique<FullPrediction>(numPredictions);
                    }
                }

                std::copy(scoreVector.scores_cbegin(), scoreVector.scores_cend(), headPtr_->scores_begin());
                headPtr_->overallQualityScore = overallQualityScore;
                return headPtr_.get();
            }

            return nullptr;
        }

    public:

        const AbstractEvaluatedPrediction* processScores(
                const AbstractEvaluatedPrediction* bestHead,
                const DenseScoreVector<FullIndexVector>& scoreVector) override {
            return processScoresInternally<DenseScoreVector<FullIndexVector>>(bestHead, scoreVector);
        }

        const AbstractEvaluatedPrediction* processScores(
                const AbstractEvaluatedPrediction* bestHead,
                const DenseScoreVector<PartialIndexVector>& scoreVector) override {
            return processScoresInternally<DenseScoreVector<PartialIndexVector>>(bestHead, scoreVector);
        }

        const AbstractEvaluatedPrediction* findHead(const AbstractEvaluatedPrediction* bestHead,
                                                    IStatisticsSubset& statisticsSubset, bool uncovered,
                                                    bool accumulated) override {
            const IScoreVector& scoreVector = statisticsSubset.calculateExampleWisePrediction(uncovered, accumulated);
            return scoreVector.processScores(bestHead, *this);
        }

        std::unique_ptr<AbstractEvaluatedPrediction> pollHead() override {
            return std::move(headPtr_);
        }

        const IScoreVector& calculatePrediction(IStatisticsSubset& statisticsSubset, bool uncovered,
                                                bool accumulated) const override {
            return statisticsSubset.calculateExampleWisePrediction(uncovered, accumulated);
        }

};

std::unique_ptr<IHeadRefinement> FullHeadRefinementFactory::create(const FullIndexVector& labelIndices) const {
    return std::make_unique<FullHeadRefinement>();
}

std::unique_ptr<IHeadRefinement> FullHeadRefinementFactory::create(const PartialIndexVector& labelIndices) const {
    return std::make_unique<FullHeadRefinement>();
}
