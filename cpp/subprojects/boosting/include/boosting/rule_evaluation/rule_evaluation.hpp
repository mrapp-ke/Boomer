/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_evaluation/score_vector.hpp"
#include <memory>


namespace boosting {

    /**
     * Defines an interface for all classes that allow to calculate the predictions of rule, as well as corresponding
     * quality scores, based on the gradients and Hessians that have been calculated according to a loss function.
     *
     * @tparam StatisticVector The type of the vector that provides access to the gradients and Hessians
     */
    template<typename StatisticVector>
    class IRuleEvaluation {

        public:

            virtual ~IRuleEvaluation() { };

            /**
             * Calculates the scores to be predicted by a rule, as well as an overall quality score, based on the sums
             * of gradients and Hessians that are covered by the rule.
             *
             * @param statisticVector   A reference to an object of template type `StatisticVector` that stores the
             *                          gradients and Hessians
             * @return                  A reference to an object of type `IScoreVector` that stores the predicted
             *                          scores, as well as an overall quality score
             */
            virtual const IScoreVector& calculatePrediction(StatisticVector& statisticVector) = 0;

    };

}
