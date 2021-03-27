#include "seco/rule_evaluation/rule_evaluation_label_wise_heuristic.hpp"
#include "seco/heuristics/confusion_matrices.hpp"
#include "common/rule_evaluation/score_vector_label_wise_dense.hpp"


namespace seco {

    /**
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, such that they optimize a
     * heuristic that is applied using label-wise averaging.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<class T>
    class HeuristicLabelWiseRuleEvaluation final : public ILabelWiseRuleEvaluation {

        private:

            std::shared_ptr<IHeuristic> heuristicPtr_;

            bool predictMajority_;

            DenseLabelWiseScoreVector<T> scoreVector_;

        public:

            /**
             * @param labelIndices      A reference to an object of template type `T` that provides access to the
             *                          indices of the labels for which the rules may predict
             * @param heuristicPtr      A shared pointer to an object of type `IHeuristic`, representing the heuristic
             *                          to be optimized
             * @param predictMajority   True, if for each label the majority label should be predicted, false, if the
             *                          minority label should be predicted
             */
            HeuristicLabelWiseRuleEvaluation(const T& labelIndices, std::shared_ptr<IHeuristic> heuristicPtr,
                                             bool predictMajority)
                : heuristicPtr_(heuristicPtr), predictMajority_(predictMajority),
                  scoreVector_(DenseLabelWiseScoreVector<T>(labelIndices)) {

            }

            const ILabelWiseScoreVector& calculateLabelWisePrediction(const uint8* minorityLabels,
                                                                      const float64* confusionMatricesTotal,
                                                                      const float64* confusionMatricesSubset,
                                                                      const float64* confusionMatricesCovered,
                                                                      bool uncovered) override {
                uint32 numPredictions = scoreVector_.getNumElements();
                typename DenseLabelWiseScoreVector<T>::score_iterator scoreIterator = scoreVector_.scores_begin();
                typename DenseLabelWiseScoreVector<T>::index_const_iterator indexIterator =
                    scoreVector_.indices_cbegin();
                typename DenseLabelWiseScoreVector<T>::quality_score_iterator qualityScoreIterator =
                    scoreVector_.quality_scores_begin();
                float64 overallQualityScore = 0;

                for (uint32 c = 0; c < numPredictions; c++) {
                    uint32 l = indexIterator[c];

                    // Set the score to be predicted for the current label...
                    uint8 minorityLabel = minorityLabels[l];
                    float64 score = (float64) (predictMajority_ ? (minorityLabel > 0 ? 0 : 1) : minorityLabel);
                    scoreIterator[c] = score;

                    // Calculate the quality score for the current label...
                    uint32 offsetC = c * NUM_CONFUSION_MATRIX_ELEMENTS;
                    uint32 offsetL = l * NUM_CONFUSION_MATRIX_ELEMENTS;
                    uint32 uin, uip, urn, urp;

                    uint32 cin = confusionMatricesCovered[offsetC + IN];
                    uint32 cip = confusionMatricesCovered[offsetC + IP];
                    uint32 crn = confusionMatricesCovered[offsetC + RN];
                    uint32 crp = confusionMatricesCovered[offsetC + RP];

                    if (uncovered) {
                        uin = cin + confusionMatricesTotal[offsetL + IN] - confusionMatricesSubset[offsetL + IN];
                        uip = cip + confusionMatricesTotal[offsetL + IP] - confusionMatricesSubset[offsetL + IP];
                        urn = crn + confusionMatricesTotal[offsetL + RN] - confusionMatricesSubset[offsetL + RN];
                        urp = crp + confusionMatricesTotal[offsetL + RP] - confusionMatricesSubset[offsetL + RP];
                        cin = confusionMatricesSubset[offsetC + IN] - cin;
                        cip = confusionMatricesSubset[offsetC + IP] - cip;
                        crn = confusionMatricesSubset[offsetC + RN] - crn;
                        crp = confusionMatricesSubset[offsetC + RP] - crp;
                    } else {
                        uin = confusionMatricesTotal[offsetL + IN] - cin;
                        uip = confusionMatricesTotal[offsetL + IP] - cip;
                        urn = confusionMatricesTotal[offsetL + RN] - crn;
                        urp = confusionMatricesTotal[offsetL + RP] - crp;
                    }

                    score = heuristicPtr_->evaluateConfusionMatrix(cin, cip, crn, crp, uin, uip, urn, urp);
                    qualityScoreIterator[c] = score;
                    overallQualityScore += score;
                }

                overallQualityScore /= numPredictions;
                scoreVector_.overallQualityScore = overallQualityScore;
                return scoreVector_;
            }

    };

    HeuristicLabelWiseRuleEvaluationFactory::HeuristicLabelWiseRuleEvaluationFactory(
            std::shared_ptr<IHeuristic> heuristicPtr, bool predictMajority)
        : heuristicPtr_(heuristicPtr), predictMajority_(predictMajority) {

    }

    std::unique_ptr<ILabelWiseRuleEvaluation> HeuristicLabelWiseRuleEvaluationFactory::create(
            const FullIndexVector& indexVector) const {
        return std::make_unique<HeuristicLabelWiseRuleEvaluation<FullIndexVector>>(indexVector, heuristicPtr_,
                                                                                   predictMajority_);
    }

    std::unique_ptr<ILabelWiseRuleEvaluation> HeuristicLabelWiseRuleEvaluationFactory::create(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<HeuristicLabelWiseRuleEvaluation<PartialIndexVector>>(indexVector, heuristicPtr_,
                                                                                      predictMajority_);
    }

}
