#include "boosting/rule_evaluation/rule_evaluation_example_wise_complete.hpp"
#include "boosting/math/math.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"
#include "common/validation.hpp"
#include "rule_evaluation_example_wise_complete_common.hpp"


namespace boosting {

    static inline void addRegularizationWeight(float64* coefficients, uint32 numPredictions,
                                               float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < numPredictions; i++) {
            coefficients[(i * numPredictions) + i] += l2RegularizationWeight;
        }
    }

    /**
     * Allows to calculate the predictions of complete rules, as well as an overall quality score, based on the
     * gradients and Hessians that are stored by a `DenseExampleWiseStatisticVector` using L2 regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<typename T>
    class ExampleWiseCompleteRuleEvaluation final :
            public AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, T> {

        private:

            DenseScoreVector<T> scoreVector_;

            float64 l2RegularizationWeight_;

            const Blas& blas_;

            const Lapack& lapack_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             * @param blas                      A reference to an object of type `Blas` that allows to execute different
             *                                  BLAS routines
             * @param lapack                    A reference to an object of type `Lapack` that allows to execute
             *                                  different LAPACK routines
             */
            ExampleWiseCompleteRuleEvaluation(const T& labelIndices, float64 l2RegularizationWeight, const Blas& blas,
                                              const Lapack& lapack)
                : AbstractExampleWiseRuleEvaluation<DenseExampleWiseStatisticVector, T>(labelIndices.getNumElements(),
                                                                                        lapack),
                  scoreVector_(DenseScoreVector<T>(labelIndices)), l2RegularizationWeight_(l2RegularizationWeight),
                  blas_(blas), lapack_(lapack) {

            }

            const IScoreVector& calculatePrediction(DenseExampleWiseStatisticVector& statisticVector) override {
                uint32 numPredictions = scoreVector_.getNumElements();

                // Copy Hessians to the matrix of coefficients and add regularization weight to its diagonal...
                copyCoefficients(statisticVector.hessians_cbegin(), this->dsysvTmpArray1_, numPredictions);
                addRegularizationWeight(this->dsysvTmpArray1_, numPredictions, l2RegularizationWeight_);

                // Copy gradients to the vector of ordinates...
                typename DenseScoreVector<T>::score_iterator scoreIterator = scoreVector_.scores_begin();
                copyOrdinates(statisticVector.gradients_cbegin(), scoreIterator, numPredictions);

                // Calculate the scores to be predicted for individual labels by solving a system of linear equations...
                lapack_.dsysv(this->dsysvTmpArray1_, this->dsysvTmpArray2_, this->dsysvTmpArray3_, scoreIterator,
                              numPredictions, this->dsysvLwork_);

                // Calculate the overall quality score...
                float64 qualityScore = calculateOverallQualityScore(scoreIterator, statisticVector.gradients_begin(),
                                                                    statisticVector.hessians_begin(),
                                                                    this->dspmvTmpArray_, numPredictions, blas_);

                // Evaluate regularization term...
                float64 regularizationTerm = 0.5 * l2RegularizationWeight_ * l2NormPow(scoreIterator, numPredictions);

                scoreVector_.overallQualityScore = qualityScore + regularizationTerm;
                return scoreVector_;
            }

    };

    ExampleWiseCompleteRuleEvaluationFactory::ExampleWiseCompleteRuleEvaluationFactory(
            float64 l2RegularizationWeight, std::unique_ptr<Blas> blasPtr, std::unique_ptr<Lapack> lapackPtr)
        : l2RegularizationWeight_(l2RegularizationWeight), blasPtr_(std::move(blasPtr)),
          lapackPtr_(std::move(lapackPtr)) {
        assertGreaterOrEqual<float64>("l2RegularizationWeight", l2RegularizationWeight, 0);
        assertNotNull("blasPtr", blasPtr_.get());
        assertNotNull("lapackPtr", lapackPtr_.get());
    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseCompleteRuleEvaluationFactory::createDense(
            const CompleteIndexVector& indexVector) const {
        return std::make_unique<ExampleWiseCompleteRuleEvaluation<CompleteIndexVector>>(indexVector,
                                                                                        l2RegularizationWeight_,
                                                                                        *blasPtr_, *lapackPtr_);
    }

    std::unique_ptr<IRuleEvaluation<DenseExampleWiseStatisticVector>> ExampleWiseCompleteRuleEvaluationFactory::createDense(
            const PartialIndexVector& indexVector) const {
        return std::make_unique<ExampleWiseCompleteRuleEvaluation<PartialIndexVector>>(indexVector,
                                                                                       l2RegularizationWeight_,
                                                                                       *blasPtr_, *lapackPtr_);
    }

}
