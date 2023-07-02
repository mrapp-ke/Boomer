/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/data/statistic_vector_label_wise_dense.hpp"
#include "common/data/arrays.hpp"
#include "common/rule_evaluation/score_vector_binned_dense.hpp"
#include "rule_evaluation_label_wise_common.hpp"

namespace boosting {

    /**
     * Calculates the score to be predicted for individual bins and returns the overall quality of the predictions.
     *
     * @tparam ScoreIterator            The type of the iterator that provides access to the gradients and Hessians
     * @param statisticIterator         An iterator that provides random access to the gradients and Hessians
     * @param scoreIterator             An iterator, the calculated scores should be written to
     * @param weights                   An iterator that provides access to the weights of individual bins
     * @param numElements               The number of bins
     * @param l1RegularizationWeight    The L1 regularization weight
     * @param l2RegularizationWeight    The L2 regularization weight
     * @return                          The overall quality that has been calculated
     */
    template<typename ScoreIterator>
    static inline float64 calculateBinnedScores(DenseLabelWiseStatisticVector::const_iterator statisticIterator,
                                                ScoreIterator scoreIterator, const uint32* weights, uint32 numElements,
                                                float64 l1RegularizationWeight, float64 l2RegularizationWeight) {
        float64 quality = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 weight = weights[i];
            const Tuple<float64>& tuple = statisticIterator[i];
            float64 predictedScore = calculateLabelWiseScore(tuple.first, tuple.second, weight * l1RegularizationWeight,
                                                             weight * l2RegularizationWeight);
            scoreIterator[i] = predictedScore;
            quality += calculateLabelWiseQuality(predictedScore, tuple.first, tuple.second,
                                                 weight * l1RegularizationWeight, weight * l2RegularizationWeight);
        }

        return quality;
    }

    /**
     * An abstract base class for all classes that allow to calculate the predictions of rules, as well as their overall
     * quality, based on the gradients and Hessians that have been calculated according to a loss function that is
     * applied label-wise and using gradient-based label binning.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the labels for which predictions should
     *                          be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class AbstractLabelWiseBinnedRuleEvaluation : public IRuleEvaluation<StatisticVector> {
        private:

            const uint32 maxBins_;

            DenseBinnedScoreVector<IndexVector> scoreVector_;

            DenseLabelWiseStatisticVector aggregatedStatisticVector_;

            uint32* numElementsPerBin_;

            float64* criteria_;

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

            const std::unique_ptr<ILabelBinning> binningPtr_;

        protected:

            /**
             * Must be implemented by subclasses in order to calculate label-wise criteria that are used to determine
             * the mapping from labels to bins.
             *
             * @param statisticVector           A reference to an object of template type `StatisticVector` that stores
             *                                  the gradients and Hessians
             * @param criteria                  A pointer to an array of type `float64`, shape `(numCriteria)`, the
             *                                  label-wise criteria should be written to
             * @param numCriteria               The number of label-wise criteria to be calculated
             * @param l1RegularizationWeight    The L1 regularization weight
             * @param l2RegularizationWeight    The L2 regularization weight
             * @return                          The number of label-wise criteria that have been calculated
             */
            virtual uint32 calculateLabelWiseCriteria(const StatisticVector& statisticVector, float64* criteria,
                                                      uint32 numCriteria, float64 l1RegularizationWeight,
                                                      float64 l2RegularizationWeight) = 0;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the labels for which the rules may predict
             * @param indicesSorted             True, if the given indices are guaranteed to be sorted, false otherwise
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             */
            AbstractLabelWiseBinnedRuleEvaluation(const IndexVector& labelIndices, bool indicesSorted,
                                                  float64 l1RegularizationWeight, float64 l2RegularizationWeight,
                                                  std::unique_ptr<ILabelBinning> binningPtr)
                : maxBins_(binningPtr->getMaxBins(labelIndices.getNumElements())),
                  scoreVector_(DenseBinnedScoreVector<IndexVector>(labelIndices, maxBins_ + 1, indicesSorted)),
                  aggregatedStatisticVector_(DenseLabelWiseStatisticVector(maxBins_)),
                  numElementsPerBin_(new uint32[maxBins_]), criteria_(new float64[labelIndices.getNumElements()]),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
                  binningPtr_(std::move(binningPtr)) {
                // The last bin is used for labels for which the corresponding criterion is zero. For this particular
                // bin, the prediction is always zero.
                scoreVector_.scores_binned_begin()[maxBins_] = 0;
            }

            virtual ~AbstractLabelWiseBinnedRuleEvaluation() override {
                delete[] numElementsPerBin_;
                delete[] criteria_;
            }

            /**
             * @see `IRuleEvaluation::evaluate`
             */
            const IScoreVector& calculateScores(StatisticVector& statisticVector) override final {
                // Calculate label-wise criteria...
                uint32 numCriteria =
                  this->calculateLabelWiseCriteria(statisticVector, criteria_, scoreVector_.getNumElements(),
                                                   l1RegularizationWeight_, l2RegularizationWeight_);

                // Obtain information about the bins to be used...
                LabelInfo labelInfo = binningPtr_->getLabelInfo(criteria_, numCriteria);
                uint32 numBins = labelInfo.numPositiveBins + labelInfo.numNegativeBins;
                scoreVector_.setNumBins(numBins, false);

                // Reset arrays to zero...
                DenseLabelWiseStatisticVector::iterator aggregatedStatisticIterator =
                  aggregatedStatisticVector_.begin();
                setArrayToZeros(aggregatedStatisticIterator, numBins);
                setArrayToZeros(numElementsPerBin_, numBins);

                // Apply binning method in order to aggregate the gradients and Hessians that belong to the same bins...
                typename StatisticVector::const_iterator statisticIterator = statisticVector.cbegin();
                typename DenseBinnedScoreVector<IndexVector>::index_binned_iterator binIndexIterator =
                  scoreVector_.indices_binned_begin();
                auto callback = [=](uint32 binIndex, uint32 labelIndex) {
                    aggregatedStatisticIterator[binIndex] += statisticIterator[labelIndex];
                    numElementsPerBin_[binIndex] += 1;
                    binIndexIterator[labelIndex] = binIndex;
                };
                auto zeroCallback = [=](uint32 labelIndex) {
                    binIndexIterator[labelIndex] = maxBins_;
                };
                binningPtr_->createBins(labelInfo, criteria_, numCriteria, callback, zeroCallback);

                // Compute predictions, as well as their overall quality...
                typename DenseBinnedScoreVector<IndexVector>::score_binned_iterator scoreIterator =
                  scoreVector_.scores_binned_begin();
                scoreVector_.quality =
                  calculateBinnedScores(aggregatedStatisticIterator, scoreIterator, numElementsPerBin_, numBins,
                                        l1RegularizationWeight_, l2RegularizationWeight_);
                return scoreVector_;
            }
    };

    /**
     * Allows to calculate the predictions of complete rules, as well as their overall quality, based on the gradients
     * Hessians that are stored by a vector using L1 and L2 regularization. The labels are assigned to bins based on the
     * gradients and Hessians.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the labels for which predictions should
     *                          be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class LabelWiseCompleteBinnedRuleEvaluation final
        : public AbstractLabelWiseBinnedRuleEvaluation<StatisticVector, IndexVector> {
        protected:

            uint32 calculateLabelWiseCriteria(const StatisticVector& statisticVector, float64* criteria,
                                              uint32 numCriteria, float64 l1RegularizationWeight,
                                              float64 l2RegularizationWeight) override {
                typename StatisticVector::const_iterator statisticIterator = statisticVector.cbegin();

                for (uint32 i = 0; i < numCriteria; i++) {
                    const Tuple<float64>& tuple = statisticIterator[i];
                    criteria[i] = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight,
                                                          l2RegularizationWeight);
                }

                return numCriteria;
            }

        public:

            /**
             * @param labelIndices              A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the labels for which the rules may predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             */
            LabelWiseCompleteBinnedRuleEvaluation(const IndexVector& labelIndices, float64 l1RegularizationWeight,
                                                  float64 l2RegularizationWeight,
                                                  std::unique_ptr<ILabelBinning> binningPtr)
                : AbstractLabelWiseBinnedRuleEvaluation<StatisticVector, IndexVector>(
                  labelIndices, true, l1RegularizationWeight, l2RegularizationWeight, std::move(binningPtr)) {}
    };

}
