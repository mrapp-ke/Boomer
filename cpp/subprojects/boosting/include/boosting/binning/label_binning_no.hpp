/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/binning/label_binning.hpp"
#include "boosting/rule_evaluation/regularization.hpp"


namespace boosting {

    /**
     * Allows to configure a method that does not assign labels to bins.
     */
    class NoLabelBinningConfig final : public ILabelBinningConfig {

        private:

            const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr_;

            const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr_;

        public:

            /**
             * @param l1RegularizationConfigPtr A reference to an unique pointer that stores the configuration of the L1
             *                                  regularization
             * @param l2RegularizationConfigPtr A reference to an unique pointer that stores the configuration of the L2
             *                                  regularization
             */
            NoLabelBinningConfig(const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
                                 const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr);

            std::unique_ptr<ILabelWiseRuleEvaluationFactory> createLabelWiseRuleEvaluationFactory() const override;

            std::unique_ptr<IExampleWiseRuleEvaluationFactory> createExampleWiseRuleEvaluationFactory(
                const Blas& blas, const Lapack& lapack) const override;

    };

}
