/**
 * Implements classes that provide access to the data that is provided for training.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "sparse.h"
#include <memory>


/**
 * An abstract base class for all label matrix that provide access to the labels of the training examples.
 */
class AbstractLabelMatrix {

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         */
        AbstractLabelMatrix(uint32 numExamples, uint32 numLabels);

        virtual ~AbstractLabelMatrix();

        /**
         * The number of examples.
         */
        uint32 numExamples_;

        /**
         * The number of labels.
         */
        uint32 numLabels_;

};

/**
 * An abstract base class for all label matrices that provide random access to the labels of the training examples.
 */
class AbstractRandomAccessLabelMatrix : public AbstractLabelMatrix {

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         */
        AbstractRandomAccessLabelMatrix(uint32 numExamples, uint32 numLabels);

        /**
         * Returns whether a specific label of the example at a given index is relevant or irrelevant.
         *
         * @param exampleIndex  The index of the example
         * @param labelIndex    The index of the label
         * @return              1, if the label is relevant, 0 otherwise
         */
        virtual uint8 getLabel(uint32 exampleIndex, uint32 labelIndex);

};

/**
 * Implements random access to the labels of the training examples based on a C-contiguous array.
 */
class DenseLabelMatrixImpl : public AbstractRandomAccessLabelMatrix {

    private:

        const uint8* y_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         * @param y             A pointer to a C-contiguous array of type `uint8`, shape `(numExamples, numLabels)`,
         *                      representing the labels of the training examples
         */
        DenseLabelMatrixImpl(uint32 numExamples, uint32 numLabels, const uint8* y);

        ~DenseLabelMatrixImpl();

        uint8 getLabel(uint32 exampleIndex, uint32 labelIndex) override;

};

/**
 * Implements random access to the labels of the training examples based on a sparse matrix in the dictionary of keys
 * (DOK) format.
 */
class DokLabelMatrixImpl : public AbstractRandomAccessLabelMatrix {

    private:

        std::shared_ptr<BinaryDokMatrix> dokMatrixPtr_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numLabels     The number of labels
         * @param dokMatrixPtr  A shared pointer to an object of type `BinaryDokMatrix`, storing the relevant labels of
         *                      the training examples
         */
        DokLabelMatrixImpl(uint32 numExamples, uint32 numLabels, std::shared_ptr<BinaryDokMatrix> dokMatrixPtr);

        ~DokLabelMatrixImpl();

        uint8 getLabel(uint32 exampleIndex, uint32 labelIndex) override;

};
