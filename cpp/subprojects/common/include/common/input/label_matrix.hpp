/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * Defines an interface for all label matrices that provide access to the labels of the training examples.
 */
class ILabelMatrix {

    public:

        virtual ~ILabelMatrix() { };

        /**
         * Returns the number of available examples.
         *
         * @return The number of examples
         */
        virtual uint32 getNumRows() const = 0;

        /**
         * Returns the number of available labels.
         *
         * @return The number of labels
         */
        virtual uint32 getNumCols() const = 0;

};

/**
 * Defines an interface for all label matrices that provide random access to the labels of the training examples.
 */
class IRandomAccessLabelMatrix : public ILabelMatrix {

    public:

        virtual ~IRandomAccessLabelMatrix() { };

        /**
         * Returns the value of a specific label.
         *
         * @param exampleIndex  The index of the example
         * @param labelIndex    The index of the label
         * @return              The value of the label
         */
        virtual uint8 getValue(uint32 exampleIndex, uint32 labelIndex) const = 0;

};
