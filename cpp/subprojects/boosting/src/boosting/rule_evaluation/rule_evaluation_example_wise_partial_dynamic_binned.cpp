#include "boosting/rule_evaluation/rule_evaluation_example_wise_partial_dynamic_binned.hpp"

#include "rule_evaluation_example_wise_binned_common.hpp"
#include "rule_evaluation_example_wise_partial_dynamic_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a subset of the available labels that is
     * determined dynamically, as well as their overall quality, based on the gradients and Hessians that are stored by
     * a `DenseExampleWiseStatisticVector` using L1 and L2 regularization. The labels are assigned to bins based on the
     * gradients and Hessians.
     *
     * @tparam IndexVector The type of the vector that provides access to the labels for which predictions should be
     *                     calculated
     */
    template<typename IndexVector>
    class DenseExampleWiseDynamicPartialBinnedRuleEvaluation final
        : public AbstractExampleWiseBinnedRuleEvaluation<DenseExampleWiseStatisticVector, PartialIndexVector> {
        private:

            const IndexVector& labelIndices_;

            const std::unique_ptr<PartialIndexVector> indexVectorPtr_;

            const float64 threshold_;

            const float64 exponent_;

        protected:

            uint32 calculateLabelWiseCriteria(const DenseExampleWiseStatisticVector& statisticVector, float64* criteria,
                                              uint32 numCriteria, float64 l1RegularizationWeight,
                                              float64 l2RegularizationWeight) override {
                uint32 numLabels = statisticVector.getNumElements();
                DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator =
                  statisticVector.gradients_cbegin();
                DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessianIterator =
                  statisticVector.hessians_diagonal_cbegin();

                const std::pair<float64, float64> pair =
                  getMinAndMaxScore(criteria, gradientIterator, hessianIterator, numLabels, l1RegularizationWeight,
                                    l2RegularizationWeight);
                float64 minAbsScore = pair.first;
                float64 threshold = calculateThreshold(minAbsScore, pair.second, threshold_, exponent_);
                PartialIndexVector::iterator indexIterator = indexVectorPtr_->begin();
                typename IndexVector::const_iterator labelIndexIterator = labelIndices_.cbegin();
                uint32 n = 0;

                for (uint32 i = 0; i < numLabels; i++) {
                    float64 score = criteria[i];

                    if (calculateWeightedScore(score, minAbsScore, exponent_) > threshold) {
                        indexIterator[n] = labelIndexIterator[i];
                        criteria[n] = score;
                        n++;
                    }
                }

                indexVectorPtr_->setNumElements(n, false);
                return n;
            }

        public:

            /**
             * @param labelIndices              A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the labels for which the rules may predict
             * @param maxBins                   The maximum number of bins
             * @param indexVectorPtr            An unique pointer to an object of type `PartialIndexVector` that stores
             *                                  the indices of the labels for which a rule predicts
             * @param threshold                 A threshold that affects for how many labels the rule heads should
             *                                  predict
             * @param exponent                  An exponent that is used to weight the estimated predictive quality for
             *                                  individual labels
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param binningPtr                An unique pointer to an object of type `ILabelBinning` that should be
             *                                  used to assign labels to bins
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute LAPACK
             *                                  routines
             */
            DenseExampleWiseDynamicPartialBinnedRuleEvaluation(
              const IndexVector& labelIndices, uint32 maxBins, std::unique_ptr<PartialIndexVector> indexVectorPtr,
              float32 threshold, float32 exponent, float64 l1RegularizationWeight, float64 l2RegularizationWeight,
              std::unique_ptr<ILabelBinning> binningPtr, const Blas& blas, const Lapack& lapack)
                : AbstractExampleWiseBinnedRuleEvaluation<DenseExampleWiseStatisticVector, PartialIndexVector>(
                  *indexVectorPtr, true, maxBins, l1RegularizationWeight, l2RegularizationWeight, std::move(binningPtr),
                  blas, lapack),
                  labelIndices_(labelIndices), indexVectorPtr_(std::move(indexVectorPtr)), threshold_(1.0 - threshold),
                  exponent_(exponent) {}
    };

    ExampleWiseDynamicPartialBinnedRuleEvaluationFactory::ExampleWiseDynamicPartialBinnedRuleEvaluationFactory(
      float32 threshold, float32 exponent, float64 l1RegularizationWeight, float64 l2RegularizationWeight,
      std::unique_ptr<ILabelBinningFactory> labelBinningFactoryPtr, const Blas& blas, const Lapack& lapack)
        : threshold_(threshold), exponent_(exponent), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight), labelBinningFactoryPtr_(std::move(labelBinningFactoryPtr)),
          blas_(blas), lapack_(lapack) {}

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>>
      ExampleWiseDynamicPartialBinnedRuleEvaluationFactory::create(
        const DenseExampleWiseStatisticVector& statisticVector, const CompleteIndexVector& indexVector) const {
        uint32 numElements = indexVector.getNumElements();
        std::unique_ptr<PartialIndexVector> indexVectorPtr = std::make_unique<PartialIndexVector>(numElements);
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        uint32 maxBins = labelBinningPtr->getMaxBins(numElements);
        return std::make_unique<DenseExampleWiseDynamicPartialBinnedRuleEvaluation<CompleteIndexVector>>(
          indexVector, maxBins, std::move(indexVectorPtr), threshold_, exponent_, l1RegularizationWeight_,
          l2RegularizationWeight_, std::move(labelBinningPtr), blas_, lapack_);
    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>>
      ExampleWiseDynamicPartialBinnedRuleEvaluationFactory::create(
        const DenseExampleWiseStatisticVector& statisticVector, const PartialIndexVector& indexVector) const {
        std::unique_ptr<ILabelBinning> labelBinningPtr = labelBinningFactoryPtr_->create();
        uint32 maxBins = labelBinningPtr->getMaxBins(indexVector.getNumElements());
        return std::make_unique<DenseExampleWiseCompleteBinnedRuleEvaluation<PartialIndexVector>>(
          indexVector, maxBins, l1RegularizationWeight_, l2RegularizationWeight_, std::move(labelBinningPtr), blas_,
          lapack_);
    }

}
