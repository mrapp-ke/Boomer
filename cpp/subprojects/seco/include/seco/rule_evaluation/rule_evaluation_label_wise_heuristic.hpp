/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "seco/rule_evaluation/rule_evaluation_label_wise.hpp"
#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Allows to create instances of the class `RegularizedLabelWiseRuleEvaluation`.
     */
    class HeuristicLabelWiseRuleEvaluationFactory final : public ILabelWiseRuleEvaluationFactory {

        private:

            std::shared_ptr<IHeuristic> heuristicPtr_;

            bool predictMajority_;

        public:

            /**
             * @param heuristicPtr      A shared pointer to an object of type `IHeuristic`, representing the heuristic
             *                          to be optimized
             * @param predictMajority   True, if for each label the majority label should be predicted, false, if the
             *                          minority label should be predicted
             */
            HeuristicLabelWiseRuleEvaluationFactory(std::shared_ptr<IHeuristic> heuristicPtr, bool predictMajority);

            std::unique_ptr<ILabelWiseRuleEvaluation> create(const FullIndexVector& indexVector) const override;

            std::unique_ptr<ILabelWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const override;

    };

}
