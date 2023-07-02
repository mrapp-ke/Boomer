#include "boosting/rule_evaluation/rule_evaluation_example_wise_partial_dynamic.hpp"

#include "rule_evaluation_example_wise_complete_common.hpp"
#include "rule_evaluation_example_wise_partial_common.hpp"
#include "rule_evaluation_example_wise_partial_dynamic_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a subset of the available labels that is
     * determined dynamically, as well as their overall quality, based on the gradients and Hessians that are stored by
     * a `DenseExampleWiseStatisticVector` using L1 and L2 regularization.
     *
     * @tparam IndexVector The type of the vector that provides access to the labels for which predictions should be
     *                     calculated
     */
    template<typename IndexVector>
    class DenseExampleWiseDynamicPartialRuleEvaluation final
        : public AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, IndexVector> {
        private:

            const IndexVector& labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            const float64 threshold_;

            const float64 exponent_;

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

            const Blas& blas_;

            const Lapack& lapack_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the labels for which the rules may predict
             * @param threshold                 A threshold that affects for how many labels the rule heads should
             *                                  predict
             * @param exponent                  An exponent that is used to weigh the estimated predictive quality for
             *                                  individual labels
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute LAPACK
             *                                  routines
             */
            DenseExampleWiseDynamicPartialRuleEvaluation(const IndexVector& labelIndices, float32 threshold,
                                                         float32 exponent, float64 l1RegularizationWeight,
                                                         float64 l2RegularizationWeight, const Blas& blas,
                                                         const Lapack& lapack)
                : AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, IndexVector>(
                  labelIndices.getNumElements(), lapack),
                  labelIndices_(labelIndices), indexVector_(PartialIndexVector(labelIndices.getNumElements())),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_, true)), threshold_(1.0 - threshold),
                  exponent_(exponent), l1RegularizationWeight_(l1RegularizationWeight),
                  l2RegularizationWeight_(l2RegularizationWeight), blas_(blas), lapack_(lapack) {}

            /**
             * @see `IRuleEvaluation::evaluate`
             */
            const IScoreVector& calculateScores(DenseExampleWiseStatisticVector& statisticVector) override {
                uint32 numLabels = statisticVector.getNumElements();
                DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator =
                  statisticVector.gradients_cbegin();
                DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessianIterator =
                  statisticVector.hessians_diagonal_cbegin();
                typename DenseScoreVector<IndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                const std::pair<float64, float64> pair =
                  getMinAndMaxScore(scoreIterator, gradientIterator, hessianIterator, numLabels,
                                    l1RegularizationWeight_, l2RegularizationWeight_);
                float64 minAbsScore = pair.first;

                // Copy gradients to the vector of ordinates and add the L1 regularization weight...
                float64 threshold = calculateThreshold(minAbsScore, pair.second, threshold_, exponent_);
                PartialIndexVector::iterator indexIterator = indexVector_.begin();
                typename IndexVector::const_iterator labelIndexIterator = labelIndices_.cbegin();
                uint32 n = 0;

                for (uint32 i = 0; i < numLabels; i++) {
                    float64 score = scoreIterator[i];

                    if (calculateWeightedScore(score, minAbsScore, exponent_) > threshold) {
                        indexIterator[n] = labelIndexIterator[i];
                        scoreIterator[n] = -gradientIterator[i];
                        n++;
                    }
                }

                indexVector_.setNumElements(n, false);
                addL1RegularizationWeight(scoreIterator, n, l1RegularizationWeight_);

                // Copy Hessians to the matrix of coefficients and add the L2 regularization weight to its diagonal...
                copyCoefficients(statisticVector.hessians_cbegin(), indexIterator, this->dsysvTmpArray1_, n);
                addL2RegularizationWeight(this->dsysvTmpArray1_, n, l2RegularizationWeight_);

                // Calculate the scores to be predicted for individual labels by solving a system of linear equations...
                lapack_.dsysv(this->dsysvTmpArray1_, this->dsysvTmpArray2_, this->dsysvTmpArray3_, scoreIterator, n,
                              this->dsysvLwork_);

                // Calculate the overall quality...
                float64 quality =
                  calculateOverallQuality(scoreIterator, statisticVector.gradients_begin(),
                                          statisticVector.hessians_begin(), this->dspmvTmpArray_, n, blas_);

                // Evaluate regularization term...
                quality +=
                  calculateRegularizationTerm(scoreIterator, n, l1RegularizationWeight_, l2RegularizationWeight_);

                scoreVector_.quality = quality;
                return scoreVector_;
            }
    };

    ExampleWiseDynamicPartialRuleEvaluationFactory::ExampleWiseDynamicPartialRuleEvaluationFactory(
      float32 threshold, float32 exponent, float64 l1RegularizationWeight, float64 l2RegularizationWeight,
      const Blas& blas, const Lapack& lapack)
        : threshold_(threshold), exponent_(exponent), l1RegularizationWeight_(l1RegularizationWeight),
          l2RegularizationWeight_(l2RegularizationWeight), blas_(blas), lapack_(lapack) {}

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>>
      ExampleWiseDynamicPartialRuleEvaluationFactory::create(const DenseExampleWiseStatisticVector& statisticVector,
                                                             const CompleteIndexVector& indexVector) const {
        return std::make_unique<DenseExampleWiseDynamicPartialRuleEvaluation<CompleteIndexVector>>(
          indexVector, threshold_, exponent_, l1RegularizationWeight_, l2RegularizationWeight_, blas_, lapack_);
    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>>
      ExampleWiseDynamicPartialRuleEvaluationFactory::create(const DenseExampleWiseStatisticVector& statisticVector,
                                                             const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseExampleWiseCompleteRuleEvaluation<PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blas_, lapack_);
    }

}
