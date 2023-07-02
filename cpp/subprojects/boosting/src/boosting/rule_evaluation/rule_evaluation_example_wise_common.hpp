/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/math/lapack.hpp"
#include "boosting/rule_evaluation/rule_evaluation.hpp"

namespace boosting {

    /**
     * An abstract base class for all classes that allow to calculate the predictions of rules, as well as their overall
     * quality, based on the gradients and Hessians that have been calculated according to a loss function that is
     * applied example-wise.
     *
     * @tparam StatisticVector  The type of the vector that provides access to the gradients and Hessians
     * @tparam IndexVector      The type of the vector that provides access to the labels for which predictions should
     *                          be calculated
     */
    template<typename StatisticVector, typename IndexVector>
    class AbstractExampleWiseRuleEvaluation : public IRuleEvaluation<StatisticVector> {
        protected:

            /**
             * A pointer to a temporary array that is used for executing the LAPACK routine DSPMV.
             */
            float64* dspmvTmpArray_;

            /**
             * A pointer to a temporary array that is used for executing the LAPACK routine DSYSV.
             */
            float64* dsysvTmpArray1_;

            /**
             * A pointer to a second temporary array that is used for executing the LAPACK routine DSYSV.
             */
            int* dsysvTmpArray2_;

            /**
             * The `lwork` parameter that is used for executing the LAPACK routine DSYSV.
             */
            const int dsysvLwork_;

            /**
             * A pointer to a third temporary array that is used for executing the LAPACK routine DSYSV.
             */
            double* dsysvTmpArray3_;

        public:

            /**
             * @param numPredictions    The number of labels for which the rules may predict
             * @param lapack            A reference to an object of type `Lapack` that allows to execute different
             *                          LAPACK routines
             */
            AbstractExampleWiseRuleEvaluation(uint32 numPredictions, const Lapack& lapack)
                : dspmvTmpArray_(new float64[numPredictions]),
                  dsysvTmpArray1_(new float64[numPredictions * numPredictions]),
                  dsysvTmpArray2_(new int[numPredictions]),
                  dsysvLwork_(lapack.queryDsysvLworkParameter(dsysvTmpArray1_, dspmvTmpArray_, numPredictions)),
                  dsysvTmpArray3_(new double[dsysvLwork_]) {}

            virtual ~AbstractExampleWiseRuleEvaluation() override {
                delete[] dspmvTmpArray_;
                delete[] dsysvTmpArray1_;
                delete[] dsysvTmpArray2_;
                delete[] dsysvTmpArray3_;
            }
    };

}
