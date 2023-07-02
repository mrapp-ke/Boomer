#include "boosting/rule_evaluation/rule_evaluation_example_wise_partial_fixed.hpp"

#include "rule_evaluation_example_wise_complete_common.hpp"
#include "rule_evaluation_example_wise_partial_common.hpp"
#include "rule_evaluation_example_wise_partial_fixed_common.hpp"

namespace boosting {

    /**
     * Allows to calculate the predictions of partial rules that predict for a predefined number of labels, as well as
     * their overall quality, based on the gradients and Hessians that are stored by a `DenseExampleWiseStatisticVector`
     * using L1 and L2 regularization.
     *
     * @tparam IndexVector The type of the vector that provides access to the labels for which predictions should be
     *                     calculated
     */
    template<typename IndexVector>
    class DenseExampleWiseFixedPartialRuleEvaluation final
        : public AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, IndexVector> {
        private:

            const IndexVector& labelIndices_;

            PartialIndexVector indexVector_;

            DenseScoreVector<PartialIndexVector> scoreVector_;

            const float64 l1RegularizationWeight_;

            const float64 l2RegularizationWeight_;

            const Blas& blas_;

            const Lapack& lapack_;

            SparseArrayVector<float64> tmpVector_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `IndexVector` that provides
             *                                  access to the indices of the labels for which the rules may predict
             * @param numPredictions            The number of labels for which the rules should predict
             * @param l1RegularizationWeight    The weight of the L1 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param blas                      A reference to an object of type `Blas` that allows to execute BLAS
             *                                  routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute LAPACK
             *                                  routines
             */
            DenseExampleWiseFixedPartialRuleEvaluation(const IndexVector& labelIndices, uint32 numPredictions,
                                                       float64 l1RegularizationWeight, float64 l2RegularizationWeight,
                                                       const Blas& blas, const Lapack& lapack)
                : AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, IndexVector>(numPredictions,
                                                                                                  lapack),
                  labelIndices_(labelIndices), indexVector_(PartialIndexVector(numPredictions)),
                  scoreVector_(DenseScoreVector<PartialIndexVector>(indexVector_, false)),
                  l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight),
                  blas_(blas), lapack_(lapack), tmpVector_(SparseArrayVector<float64>(labelIndices.getNumElements())) {}

            /**
             * @see `IRuleEvaluation::evaluate`
             */
            const IScoreVector& calculateScores(DenseExampleWiseStatisticVector& statisticVector) override {
                uint32 numLabels = statisticVector.getNumElements();
                uint32 numPredictions = indexVector_.getNumElements();
                DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator =
                  statisticVector.gradients_cbegin();
                DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessianIterator =
                  statisticVector.hessians_diagonal_cbegin();
                SparseArrayVector<float64>::iterator tmpIterator = tmpVector_.begin();
                sortLabelWiseCriteria(tmpIterator, gradientIterator, hessianIterator, numLabels, numPredictions,
                                      l1RegularizationWeight_, l2RegularizationWeight_);

                // Copy gradients to the vector of ordinates and add the L1 regularization weight...
                PartialIndexVector::iterator indexIterator = indexVector_.begin();
                typename DenseScoreVector<IndexVector>::score_iterator scoreIterator = scoreVector_.scores_begin();
                typename IndexVector::const_iterator labelIndexIterator = labelIndices_.cbegin();

                for (uint32 i = 0; i < numPredictions; i++) {
                    const IndexedValue<float64>& entry = tmpIterator[i];
                    uint32 index = entry.index;
                    indexIterator[i] = labelIndexIterator[index];
                    scoreIterator[i] = -gradientIterator[index];
                }

                addL1RegularizationWeight(scoreIterator, numPredictions, l1RegularizationWeight_);

                // Copy Hessians to the matrix of coefficients and add the L2 regularization weight to its diagonal...
                copyCoefficients(statisticVector.hessians_cbegin(), indexIterator, this->dsysvTmpArray1_,
                                 numPredictions);
                addL2RegularizationWeight(this->dsysvTmpArray1_, numPredictions, l2RegularizationWeight_);

                // Calculate the scores to be predicted for individual labels by solving a system of linear equations...
                lapack_.dsysv(this->dsysvTmpArray1_, this->dsysvTmpArray2_, this->dsysvTmpArray3_, scoreIterator,
                              numPredictions, this->dsysvLwork_);

                // Calculate the overall quality...
                float64 quality = calculateOverallQuality(scoreIterator, statisticVector.gradients_begin(),
                                                          statisticVector.hessians_begin(), this->dspmvTmpArray_,
                                                          numPredictions, blas_);

                // Evaluate regularization term...
                quality += calculateRegularizationTerm(scoreIterator, numPredictions, l1RegularizationWeight_,
                                                       l2RegularizationWeight_);

                scoreVector_.quality = quality;
                return scoreVector_;
            }
    };

    ExampleWiseFixedPartialRuleEvaluationFactory::ExampleWiseFixedPartialRuleEvaluationFactory(
      float32 labelRatio, uint32 minLabels, uint32 maxLabels, float64 l1RegularizationWeight,
      float64 l2RegularizationWeight, const Blas& blas, const Lapack& lapack)
        : labelRatio_(labelRatio), minLabels_(minLabels), maxLabels_(maxLabels),
          l1RegularizationWeight_(l1RegularizationWeight), l2RegularizationWeight_(l2RegularizationWeight), blas_(blas),
          lapack_(lapack) {}

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>>
      ExampleWiseFixedPartialRuleEvaluationFactory::create(const DenseExampleWiseStatisticVector& statisticVector,
                                                           const CompleteIndexVector& indexVector) const {
        uint32 numPredictions =
          calculateBoundedFraction(indexVector.getNumElements(), labelRatio_, minLabels_, maxLabels_);
        return std::make_unique<DenseExampleWiseFixedPartialRuleEvaluation<CompleteIndexVector>>(
          indexVector, numPredictions, l1RegularizationWeight_, l2RegularizationWeight_, blas_, lapack_);
    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>>
      ExampleWiseFixedPartialRuleEvaluationFactory::create(const DenseExampleWiseStatisticVector& statisticVector,
                                                           const PartialIndexVector& indexVector) const {
        return std::make_unique<DenseExampleWiseCompleteRuleEvaluation<PartialIndexVector>>(
          indexVector, l1RegularizationWeight_, l2RegularizationWeight_, blas_, lapack_);
    }

}
