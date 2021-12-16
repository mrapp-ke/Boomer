/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/post_processing/post_processor.hpp"


namespace boosting {

    /**
     * Post-processes the predictions of rules by shrinking their weights by a constant shrinkage parameter.
     */
    class ConstantShrinkage final : public IPostProcessor {

        private:

            float64 shrinkage_;

        public:

            /**
             * @param shrinkage The shrinkage parameter. Must be in (0, 1).
             */
            ConstantShrinkage(float64 shrinkage);

            /**
             * @see `IPostProcessor::postProcess`
             */
            void postProcess(AbstractPrediction& prediction) const override;

    };

}
