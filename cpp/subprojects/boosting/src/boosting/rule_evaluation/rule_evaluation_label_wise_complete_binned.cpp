#include "boosting/rule_evaluation/rule_evaluation_label_wise_complete_binned.hpp"
#include "common/rule_evaluation/score_vector_binned_dense.hpp"
#include "common/data/arrays.hpp"
#include "common/validation.hpp"
#include "rule_evaluation_label_wise_complete_common.hpp"


namespace boosting {

    template<typename ScoreIterator>
    static inline void calculateLabelWiseScores(DenseLabelWiseStatisticVector::const_iterator statisticIterator,
                                                ScoreIterator scoreIterator, const uint32* weights, uint32 numElements,
                                                float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < numElements; i++) {
            uint32 weight = weights[i];
            const Tuple<float64>& tuple = statisticIterator[i];
            scoreIterator[i] = calculateLabelWiseScore(tuple.first, tuple.second, weight * l2RegularizationWeight);
        }
    }

    template<typename ScoreIterator>
    static inline constexpr float64 calculateOverallQualityScore(
            DenseLabelWiseStatisticVector::const_iterator statisticIterator, ScoreIterator scoreIterator,
            const uint32* weights, uint32 numElements, float64 l2RegularizationWeight) {
        float64 overallQualityScore = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 weight = weights[i];
            const Tuple<float64>& tuple = statisticIterator[i];
            overallQualityScore += calculateLabelWiseQualityScore(scoreIterator[i], tuple.first, tuple.second,
                                                                  weight * l2RegularizationWeight);
        }

        return overallQualityScore;
    }

    /**
     * Allows to calculate the predictions of complete rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `DenseLabelWiseStatisticVector` using L2 regularization. The labels
     * are assigned to bins based on the gradients and Hessians.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class DenseLabelWiseCompleteBinnedRuleEvaluation final : public IRuleEvaluation<DenseLabelWiseStatisticVector> {

        private:

            uint32 maxBins_;

            DenseBinnedScoreVector<T> scoreVector_;

            DenseLabelWiseStatisticVector aggregatedStatisticVector_;

            uint32* numElementsPerBin_;

            float64* criteria_;

            float64 l2RegularizationWeight_;

            std::unique_ptr<ILabelBinning> binningPtr_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             */
            DenseLabelWiseCompleteBinnedRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight,
                                                       std::unique_ptr<ILabelBinning> binningPtr)
                : maxBins_(binningPtr->getMaxBins(labelIndices.getNumElements())),
                  scoreVector_(DenseBinnedScoreVector<T>(labelIndices, maxBins_ + 1)),
                  aggregatedStatisticVector_(DenseLabelWiseStatisticVector(maxBins_)),
                  numElementsPerBin_(new uint32[maxBins_]), criteria_(new float64[labelIndices.getNumElements()]),
                  l2RegularizationWeight_(l2RegularizationWeight), binningPtr_(std::move(binningPtr)) {
                // The last bin is used for labels for which the corresponding criterion is zero. For this particular
                // bin, the prediction is always zero.
                scoreVector_.scores_binned_begin()[maxBins_] = 0;
            }

            ~DenseLabelWiseCompleteBinnedRuleEvaluation() {
                delete[] numElementsPerBin_;
                delete[] criteria_;
            }

            const IScoreVector& calculatePrediction(DenseLabelWiseStatisticVector& statisticVector) override {
                // Calculate label-wise criteria...
                uint32 numLabels = statisticVector.getNumElements();
                DenseLabelWiseStatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                calculateLabelWiseScores(statisticIterator, criteria_, numLabels, l2RegularizationWeight_);

                // Obtain information about the bins to be used...
                LabelInfo labelInfo = binningPtr_->getLabelInfo(criteria_, numLabels);
                uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;
                scoreVector_.setNumBins(numBins, false);

                // Reset arrays to zero...
                DenseLabelWiseStatisticVector::iterator aggregatedStatisticIterator =
                    aggregatedStatisticVector_.begin();
                setArrayToZeros(aggregatedStatisticIterator, numBins);
                setArrayToZeros(numElementsPerBin_, numBins);

                // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
                typename DenseBinnedScoreVector<T>::index_binned_iterator binIndexIterator =
                    scoreVector_.indices_binned_begin();
                auto callback = [=](uint32 binIndex, uint32 labelIndex) {
                    aggregatedStatisticIterator[binIndex] += statisticIterator[labelIndex];
                    numElementsPerBin_[binIndex] += 1;
                    binIndexIterator[labelIndex] = binIndex;
                };
                auto zeroCallback = [=](uint32 labelIndex) {
                    binIndexIterator[labelIndex] = maxBins_;
                };
                binningPtr_->createBins(labelInfo, criteria_, numLabels, callback, zeroCallback);

                // Compute predictions, as well as an overall quality score...
                typename DenseBinnedScoreVector<T>::score_binned_iterator scoreIterator =
                    scoreVector_.scores_binned_begin();
                calculateLabelWiseScores(aggregatedStatisticIterator, scoreIterator, numElementsPerBin_, numBins,
                                         l2RegularizationWeight_);
                scoreVector_.overallQualityScore = calculateOverallQualityScore(aggregatedStatisticIterator,
                                                                                scoreIterator, numElementsPerBin_,
                                                                                numBins, l2RegularizationWeight_);
                return scoreVector_;
            }

    };

    LabelWiseCompleteBinnedRuleEvaluationFactory::LabelWiseCompleteBinnedRuleEvaluationFactory(
            float64 l2RegularizationWeight, std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr)
        : l2RegularizationWeight_(l2RegularizationWeight), labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)) {
        assertGreaterOrEqual<float64>("l2RegularizationWeight", l2RegularizationWeight, 0);
        assertNotNull("labelBinningFactoryPtr", labelBinningFactoryPtr_.get());
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseCompleteBinnedRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<DenseLabelWiseCompleteBinnedRuleEvaluation<CompleteIndexVector>>(
            indexVector, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

    std::unique_ptr<IRuleEvaluation<DenseLabelWiseStatisticVector>> LabelWiseCompleteBinnedRuleEvaluationFactory::create(
            const DenseLabelWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        return std::make_unique<DenseLabelWiseCompleteBinnedRuleEvaluation<PartialIndexVector>>(
            indexVector, l2RegularizationWeight_, std::move(labelBinningPtr));
    }

}