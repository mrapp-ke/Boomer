/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#include "common/data/types.hpp"


namespace seco {

    /**
     * The number of elements in a confusion matrix.
     */
    const uint32 NUM_CONFUSION_MATRIX_ELEMENTS = 4;

    /**
     * An enum that specifies all positive elements of a confusion matrix.
     */
    enum ConfusionMatrixElement : uint32 {
        IN = 0,
        IP = 1,
        RN = 2,
        RP = 3
    };

    /**
     * Returns the confusion matrix element, a label corresponds to, depending on the ground truth an a prediction.
     *
     * @param trueLabel         The true label according to the ground truth
     * @param predictedLabel    The predicted label
     * @return                  The confusion matrix element
     */
    static inline ConfusionMatrixElement getConfusionMatrixElement(uint8 trueLabel, uint8 predictedLabel) {
        if (trueLabel) {
            return predictedLabel ? RP : RN;
        } else {
            return predictedLabel ? IP : IN;
        }
    }

}
