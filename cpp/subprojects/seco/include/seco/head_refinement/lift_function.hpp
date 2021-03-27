/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke-tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


namespace seco {

    /**
     * An abstract base class for all lift functions.
     */
    class ILiftFunction {

        public:

            virtual ~ILiftFunction() { };

            /**
             * Calculates and returns the lift for a specific number of labels.
             *
             * @param numLabels The number of labels for which the lift should be calculated
             * @return          The lift that has been calculated
             */
            virtual float64 calculateLift(uint32 numLabels) const = 0;

            /**
             * Returns the maximum lift possible.
             *
             * @return The maximum lift possible
             */
            virtual float64 getMaxLift() const = 0;

    };

}
