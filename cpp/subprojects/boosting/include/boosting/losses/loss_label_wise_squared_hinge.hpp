#pragma once

#include "boosting/losses/loss_label_wise.hpp"


namespace boosting {

    /**
     * A multi-label variant of the squared hinge loss that is applied label-wise.
     */
    class LabelWiseSquaredHingeLoss final : public AbstractLabelWiseLoss {

        public:

            LabelWiseSquaredHingeLoss();

    };

}
