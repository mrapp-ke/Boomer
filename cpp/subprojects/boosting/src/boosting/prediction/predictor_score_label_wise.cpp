#include "boosting/prediction/predictor_score_label_wise.hpp"

#include "boosting/prediction/predictor_score_common.hpp"

namespace boosting {

    /**
     * Allows to create instances of the type `IScorePredictor` that predict label-wise regression scores for given
     * query examples by summing up the scores that are provided by individual rules for each label individually.
     */
    class LabelWiseScorePredictorFactory final : public IScorePredictorFactory {
        private:

            const uint32 numThreads_;

        public:

            /**
             * @param numThreads The number of CPU threads to be used to make predictions for different query examples
             *                   in parallel. Must be at least 1
             */
            LabelWiseScorePredictorFactory(uint32 numThreads) : numThreads_(numThreads) {}

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IScorePredictor> create(const CContiguousConstView<const float32>& featureMatrix,
                                                    const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                    uint32 numLabels) const override {
                return std::make_unique<ScorePredictor<CContiguousConstView<const float32>, RuleList>>(
                  featureMatrix, model, numLabels, numThreads_);
            }

            /**
             * @see `IPredictorFactory::create`
             */
            std::unique_ptr<IScorePredictor> create(const CsrConstView<const float32>& featureMatrix,
                                                    const RuleList& model, const LabelVectorSet* labelVectorSet,
                                                    uint32 numLabels) const override {
                return std::make_unique<ScorePredictor<CsrConstView<const float32>, RuleList>>(featureMatrix, model,
                                                                                               numLabels, numThreads_);
            }
    };

    LabelWiseScorePredictorConfig::LabelWiseScorePredictorConfig(
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    std::unique_ptr<IScorePredictorFactory> LabelWiseScorePredictorConfig::createPredictorFactory(
      const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const {
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, numLabels);
        return std::make_unique<LabelWiseScorePredictorFactory>(numThreads);
    }

    bool LabelWiseScorePredictorConfig::isLabelVectorSetNeeded() const {
        return false;
    }

}
