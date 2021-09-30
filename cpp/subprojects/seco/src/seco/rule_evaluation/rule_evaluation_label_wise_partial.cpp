#include "seco/rule_evaluation/rule_evaluation_label_wise_partial.hpp"
#include "common/data/vector_sparse_array.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"
#include "common/validation.hpp"
#include "rule_evaluation_label_wise_common.hpp"
#include <algorithm>


namespace seco {

    static inline float64 calculateLiftedQualityScore(float64 qualityScore, uint32 numPredictions,
                                                      const ILiftFunction& liftFunction) {
        return (qualityScore / numPredictions) * liftFunction.calculateLift(numPredictions);
    }

    /**
     * Allows to calculate the predictions of complete rules, as well as corresponding quality scores, such that they
     * optimize a heuristic that is applied using label-wise averaging and taking a specific lift function, which
     * affects the quality score of rules, depending on how many labels they predict, into account.
     */
    class LabelWiseCompleteRuleEvaluation final : public IRuleEvaluation {

        private:

            DenseScoreVector<PartialIndexVector> scoreVector_;

            const IHeuristic& heuristic_;

            const ILiftFunction& liftFunction_;

        public:

            /**
             * @param labelIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @param heuristic     A reference to an object of type `IHeuristic`, implementing the heuristic to be
             *                      optimized
             * @param liftFunction  A reference to an object of type `ILiftFunction` that should affect the quality
             *                      scores of rules, depending on how many labels they predict
             */
            LabelWiseCompleteRuleEvaluation(const PartialIndexVector& labelIndices, const IHeuristic& heuristic,
                                            const ILiftFunction& liftFunction)
                : scoreVector_(DenseScoreVector<PartialIndexVector>(labelIndices)), heuristic_(heuristic),
                  liftFunction_(liftFunction) {

            }

            const IScoreVector& calculatePrediction(
                    const BinarySparseArrayVector& majorityLabelVector,
                    const DenseConfusionMatrixVector& confusionMatricesTotal,
                    const DenseConfusionMatrixVector& confusionMatricesCovered) override {
                uint32 numElements = scoreVector_.getNumElements();
                DenseScoreVector<PartialIndexVector>::index_const_iterator indexIterator =
                    scoreVector_.indices_cbegin();
                DenseConfusionMatrixVector::const_iterator totalIterator = confusionMatricesTotal.cbegin();
                DenseConfusionMatrixVector::const_iterator coveredIterator = confusionMatricesCovered.cbegin();
                auto labelIterator = make_binary_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                                  majorityLabelVector.indices_cend());
                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                float64 sumOfQualityScores = 0;
                uint32 previousIndex = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = indexIterator[i];
                    std::advance(labelIterator, index - previousIndex);
                    scoreIterator[i] = (float64) !(*labelIterator);
                    sumOfQualityScores += (1 - calculateLabelWiseQualityScore(totalIterator[index], coveredIterator[i],
                                                                              heuristic_));
                    previousIndex = index;
                }

                scoreVector_.overallQualityScore = (1 - calculateLiftedQualityScore(sumOfQualityScores, numElements,
                                                                                    liftFunction_));
                return scoreVector_;
            }

    };

    /**
     * Allows to calculate the predictions of partial rules, as well as corresponding quality scores, such that they
     * optimize a heuristic that is applied using label-wise averaging and taking a specific lift function, which
     * affects the quality score of rules, depending on how many labels they predict, into account.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class LabelWisePartialRuleEvaluation final : public IRuleEvaluation {

        private:

            const T& labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            SparseArrayVector<float64> sortedVector_;

            const IHeuristic& heuristic_;

            const ILiftFunction& liftFunction_;

        public:

            /**
             * @param labelIndices  A reference to an object of template type `T` that provides access to the indices of
             *                      the labels for which the rules may predict
             * @param heuristic     A reference to an object of type `IHeuristic`, implementing the heuristic to be
             *                      optimized
             * @param liftFunction  A reference to an object of type `ILiftFunction` that should affect the quality
             *                      scores of rules, depending on how many labels they predict
             */
            LabelWisePartialRuleEvaluation(const T& labelIndices, const IHeuristic& heuristic,
                                           const ILiftFunction& liftFunction)
                : labelIndices_(labelIndices), indexVector_(PartialIndexVector(labelIndices.getNumElements())),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_)),
                  sortedVector_(SparseArrayVector<float64>(labelIndices.getNumElements())),
                  heuristic_(heuristic), liftFunction_(liftFunction) {

            }

            const IScoreVector& calculatePrediction(
                    const BinarySparseArrayVector& majorityLabelVector,
                    const DenseConfusionMatrixVector& confusionMatricesTotal,
                    const DenseConfusionMatrixVector& confusionMatricesCovered) override {
                uint32 numElements = labelIndices_.getNumElements();
                typename T::const_iterator indexIterator = labelIndices_.cbegin();
                DenseConfusionMatrixVector::const_iterator totalIterator = confusionMatricesTotal.cbegin();
                DenseConfusionMatrixVector::const_iterator coveredIterator = confusionMatricesCovered.cbegin();
                auto labelIterator = make_binary_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                                  majorityLabelVector.indices_cend());
                DenseScoreVector<PartialIndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                PartialIndexVector::iterator predictedIndexIterator = indexVector_.begin();
                SparseArrayVector<float64>::iterator sortedIterator = sortedVector_.begin();

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = indexIterator[i];
                    sortedIterator[i].index = index;
                    sortedIterator[i].value = calculateLabelWiseQualityScore(totalIterator[index], coveredIterator[i],
                                                                             heuristic_);
                }

                sortedVector_.sortByValues();
                float64 maxLift = liftFunction_.getMaxLift();
                float64 sumOfQualityScores = (1 - sortedIterator[0].value);
                float64 bestQualityScore = calculateLiftedQualityScore(sumOfQualityScores, 1, liftFunction_);
                uint32 bestNumPredictions = 1;

                for (uint32 i = 1; i < numElements; i++) {
                    uint32 numPredictions = i + 1;
                    sumOfQualityScores += (1 - sortedIterator[i].value);
                    float64 qualityScore = calculateLiftedQualityScore(sumOfQualityScores, numPredictions,
                                                                       liftFunction_);

                    if (qualityScore > bestQualityScore) {
                        bestNumPredictions = numPredictions;
                        bestQualityScore = qualityScore;
                    }

                    if (qualityScore * maxLift < bestQualityScore) {
                        // Prunable by decomposition...
                        break;
                    }
                }

                std::sort(sortedIterator, &sortedIterator[bestNumPredictions], [=](const IndexedValue<float64>& a,
                                                                                   const IndexedValue<float64>& b) {
                    return a.index < b.index;
                });
                indexVector_.setNumElements(bestNumPredictions, false);
                scoreVector_.overallQualityScore = (1 - bestQualityScore);
                uint32 previousIndex = 0;

                for (uint32 i = 0; i < bestNumPredictions; i++) {
                    uint32 index = sortedIterator[i].index;
                    std::advance(labelIterator, index - previousIndex);
                    scoreIterator[i] = (float64) !(*labelIterator);
                    predictedIndexIterator[i] = index;
                    previousIndex = index;
                }

                return scoreVector_;
            }

    };

    LabelWisePartialRuleEvaluationFactory::LabelWisePartialRuleEvaluationFactory(
            std::unique_ptr<IHeuristic> heuristicPtr, std::unique_ptr<ILiftFunction> liftFunctionPtr)
        : heuristicPtr_(std::move(heuristicPtr)), liftFunctionPtr_(std::move(liftFunctionPtr)) {
        assertNotNull("heuristicPtr", heuristicPtr_.get());
        assertNotNull("liftFunctionPtr", liftFunctionPtr_.get());
    }

    std::unique_ptr<IRuleEvaluation> LabelWisePartialRuleEvaluationFactory::create(
            const CompleteIndexVector& indexVector) const {
        return std::make_unique<LabelWisePartialRuleEvaluation<CompleteIndexVector>>(indexVector, *heuristicPtr_,
                                                                                     *liftFunctionPtr_);
    }

    std::unique_ptr<IRuleEvaluation> LabelWisePartialRuleEvaluationFactory::create(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWiseCompleteRuleEvaluation>(indexVector, *heuristicPtr_, *liftFunctionPtr_);
    }

}
