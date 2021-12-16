/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/rule_evaluation/rule_evaluation.hpp"
#include "boosting/math/lapack.hpp"


namespace boosting {


    /**
     * Copies Hessians from an iterator to a matrix of coefficients that may be passed to LAPACK's DSYSV routine.
     *
     * @tparam HessianIterator  The type of the iterator that provides access to the Hessians
     * @param hessianIterator   An iterator that provides random access to the Hessians
     * @param coefficients      An array of type `float64`, shape `(n * n)`, the Hessians should be copied to
     * @param n                 The dimensionality of the matrix of coefficients
     */
    template<typename HessianIterator>
    static inline void copyCoefficients(HessianIterator hessianIterator, float64* coefficients, uint32 n) {
        for (uint32 c = 0; c < n; c++) {
            uint32 offset = c * n;

            for (uint32 r = 0; r <= c; r++) {
                coefficients[offset + r] = *hessianIterator;
                hessianIterator++;
            }
        }
    }

    /**
     * Copies gradients from an iterator to a vector of ordinates that may be passed to LAPACK's DSYSV routine.
     *
     * @tparam GradientIterator         The type of the iterator that provides access to the gradients
     * @param gradientIterator          An iterator that provides random access to the gradients
     * @param ordinates                 An array of type `float64`, shape `(n)`, the gradients should be copied to
     * @param n                         The number of gradients
     */
    template<typename GradientIterator>
    static inline void copyOrdinates(GradientIterator gradientIterator, float64* ordinates, uint32 n) {
        for (uint32 i = 0; i < n; i++) {
            ordinates[i] = -gradientIterator[i];
        }
    }

    /**
     * Calculates and returns an overall quality score that assesses the quality of predictions for several labels.
     *
     * @tparam ScoreIterator    The type of the iterator that provides access to the predicted scores
     * @tparam GradientIterator The type of the iterator that provides access to the gradients
     * @tparam HessianIterator  The type of the iterator that provides access to the Hessians
     * @param scores            An iterator that provides random access to the predicted scores
     * @param gradients         An iterator that provides random access to the gradients
     * @param hessians          An iterator that provides random access to the Hessians
     * @param tmpArray          A pointer to an array of type `float64`, shape `(numPredictions)`, that should be used
     *                          by BLAS' DSPMV routine to store temporary values
     * @param numPredictions    The number of predictions
     * @param blas              A reference to an object of type `Blas` that allows to execute different BLAS routines
     * @return                  The quality score that has been calculated
     */
    template<typename ScoreIterator, typename GradientIterator, typename HessianIterator>
    static inline float64 calculateOverallQualityScore(ScoreIterator scores, GradientIterator gradients,
                                                       HessianIterator hessians, float64* tmpArray,
                                                       uint32 numPredictions, const Blas& blas) {
        blas.dspmv(hessians, scores, tmpArray, numPredictions);
        return blas.ddot(scores, gradients, numPredictions) + (0.5 * blas.ddot(scores, tmpArray, numPredictions));
    }

    /**
     * An abstract base class for all classes that allow to calculate the predictions of rules, as well as corresponding
     * quality scores, based on the gradients and Hessians that have been calculated according to a loss function that
     * is applied example-wise.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam T                The type of the vector that provides access to the labels for which predictions should
     *                          be calculated
     */
    template<typename StatisticVector, typename T>
    class AbstractExampleWiseRuleEvaluation : public IRuleEvaluation<StatisticVector> {

        protected:

            /**
             * The `lwork` parameter that is used for executing the LAPACK routine DSYSV.
             */
            int dsysvLwork_;

            /**
             * A pointer to a temporary array that is used for executing the LAPACK routine DSYSV.
             */
            float64* dsysvTmpArray1_;

            /**
             * A pointer to a second temporary array that is used for executing the LAPACK routine DSYSV.
             */
            int* dsysvTmpArray2_;

            /**
             * A pointer to a third temporary array that is used for executing the LAPACK routine DSYSV.
             */
            double* dsysvTmpArray3_;

            /**
             * A pointer to a temporary array that is used for executing the LAPACK routine DSPMV.
             */
            float64* dspmvTmpArray_;

        public:

            /**
             * @param numPredictions    The number of labels for which the rules may predict
             * @param lapack            A reference to an object of type `Lapack` that allows to execute different
             *                          LAPACK routines
             */
            AbstractExampleWiseRuleEvaluation(uint32 numPredictions, const Lapack& lapack) {
                dsysvTmpArray1_ = new float64[numPredictions * numPredictions];
                dsysvTmpArray2_ = new int[numPredictions];
                dspmvTmpArray_ = new float64[numPredictions];

                // Query the optimal "lwork" parameter to be used by LAPACK's DSYSV routine...
                dsysvLwork_ = lapack.queryDsysvLworkParameter(dsysvTmpArray1_, dspmvTmpArray_, numPredictions);
                dsysvTmpArray3_ = new double[dsysvLwork_];
            }

            virtual ~AbstractExampleWiseRuleEvaluation() {
                delete[] dsysvTmpArray1_;
                delete[] dsysvTmpArray2_;
                delete[] dsysvTmpArray3_;
                delete[] dspmvTmpArray_;
            }

    };

}
