/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss_label_wise.hpp"


namespace boosting {

    /**
     * A multi-label variant of the logistic loss that is applied label-wise.
     */
    class LabelWiseLogisticLoss final : public AbstractLabelWiseLoss {

        public:

            LabelWiseLogisticLoss();

    };

}
