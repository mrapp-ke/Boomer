#include "seco/rule_evaluation/rule_evaluation_label_wise_majority.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"


namespace seco {

    /**
     * Allows to calculate the predictions of rules, such that they predict each label as relevant or irrelevant,
     * depending on whether it is associated with the majority of the training examples or not.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class LabelWiseMajorityRuleEvaluation final : public IRuleEvaluation {

        private:

            DenseScoreVector<T> scoreVector_;

        public:

            /**
             * @param labelIndices A reference to an object of template type `T` that provides access to the indices of
             *                     the labels for which the rules may predict
             */
            LabelWiseMajorityRuleEvaluation(const T& labelIndices)
                : scoreVector_(DenseScoreVector<T>(labelIndices)) {
                scoreVector_.overallQualityScore = 0;
            }

            const IScoreVector& calculatePrediction(
                    const BinarySparseArrayVector& majorityLabelVector,
                    const DenseConfusionMatrixVector& confusionMatricesTotal,
                    const DenseConfusionMatrixVector& confusionMatricesCovered) override {
                typename DenseScoreVector<T>::score_iterator scoreIterator = scoreVector_.scores_begin();
                typename DenseScoreVector<T>::index_const_iterator indexIterator = scoreVector_.indices_cbegin();
                auto labelIterator = make_binary_forward_iterator(majorityLabelVector.indices_cbegin(),
                                                                  majorityLabelVector.indices_cend());
                uint32 numElements = scoreVector_.getNumElements();
                uint32 previousIndex = 0;

                for (uint32 i = 0; i < numElements; i++) {
                    uint32 index = indexIterator[i];
                    std::advance(labelIterator, index - previousIndex);
                    scoreIterator[i] = (float64) *labelIterator;
                    previousIndex = index;
                }

                return scoreVector_;
            }

    };

    std::unique_ptr<IRuleEvaluation> LabelWiseMajorityRuleEvaluationFactory::create(
            const CompleteIndexVector& indexVector) const {
        return std::make_unique<LabelWiseMajorityRuleEvaluation<CompleteIndexVector>>(indexVector);
    }

    std::unique_ptr<IRuleEvaluation> LabelWiseMajorityRuleEvaluationFactory::create(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<LabelWiseMajorityRuleEvaluation<PartialIndexVector>>(indexVector);
    }

}
