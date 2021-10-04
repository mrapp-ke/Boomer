/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "seco/rule_evaluation/rule_evaluation_label_wise.hpp"
#include "seco/rule_evaluation/lift_function.hpp"
#include "seco/heuristics/heuristic.hpp"


namespace seco {

    /**
     * Allows to create instances of the class `LabelWisePartialRuleEvaluation`.
     */
    class LabelWisePartialRuleEvaluationFactory final : public ILabelWiseRuleEvaluationFactory {

        private:

            std::unique_ptr<IHeuristic> heuristicPtr_;

            std::unique_ptr<ILiftFunction> liftFunctionPtr_;

        public:

            /**
             * @param heuristicPtr      An unique pointer to an object of type `IHeuristic`, implementing the heuristic
             *                          to be optimized
             * @param liftFunctionPtr   An unique pointer to an object of type `ILiftFunction` that should affect the
             *                          quality scores of rules, depending on how many labels they predict
             */
            LabelWisePartialRuleEvaluationFactory(std::unique_ptr<IHeuristic> heuristicPtr,
                                                  std::unique_ptr<ILiftFunction> liftFunctionPtr);

            std::unique_ptr<IRuleEvaluation> create(const CompleteIndexVector& indexVector) const override;

            std::unique_ptr<IRuleEvaluation> create(const PartialIndexVector& indexVector) const override;

    };

}
