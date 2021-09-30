/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "boosting/losses/loss_label_wise.hpp"


namespace boosting {

    /**
     * A multi-label variant of the squared error loss that is applied label-wise.
     */
    class LabelWiseSquaredErrorLoss final : public AbstractLabelWiseLoss {

        public:

            LabelWiseSquaredErrorLoss();

    };

}
