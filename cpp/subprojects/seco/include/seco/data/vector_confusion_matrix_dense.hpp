/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/input/label_matrix_c_contiguous.hpp"
#include "common/input/label_matrix_csr.hpp"
#include "seco/data/confusion_matrix.hpp"
#include "seco/data/matrix_weight_dense.hpp"


namespace seco {

    /**
     * An one-dimensional vector that stores a fixed number of confusion matrices in a C-contiguous array.
     */
    class DenseConfusionMatrixVector {

        private:

            ConfusionMatrix* array_;

            uint32 numElements_;

        public:

            /**
             * @param numElements The number of elements in the vector
             */
            DenseConfusionMatrixVector(uint32 numElements);

            /**
             * @param numElements   The number of elements in the vector
             * @param init          True, if the elements of all confusion matrices should be value-initialized
             */
            DenseConfusionMatrixVector(uint32 numElements, bool init);

            /**
             * @param other A reference to an object of type `DenseConfusionMatrixVector` to be copied
             */
            DenseConfusionMatrixVector(const DenseConfusionMatrixVector& other);

            ~DenseConfusionMatrixVector();

            /**
             * An iterator that provides access to the elements in a confusion matrix and allows to modify them.
             */
            typedef ConfusionMatrix* iterator;

            /**
             * An iterator that provides read-only access to the elements in a confusion matrix.
             */
            typedef const ConfusionMatrix* const_iterator;

            /**
             * Returns an `iterator` to the beginning of the vector.
             *
             * @return An `iterator` to the beginning
             */
            iterator begin();

            /**
             * Returns an `iterator` to the end of the vector.
             *
             * @return An `iterator` to the end
             */
            iterator end();

            /**
             * Returns a `const_iterator` to the beginning of the vector.
             *
             * @return A `const_iterator` to the beginning
             */
            const_iterator cbegin() const;

            /**
             * Returns a `const_iterator` to the end of the vector.
             *
             * @return A `const_iterator` to the end
             */
            const_iterator cend() const;

            /**
             * Returns the number of elements in the vector.
             *
             * @return The number of elements
             */
            uint32 getNumElements() const;

            /**
             * Sets the elements of all confusion matrices to zero.
             */
            void clear();

            /**
             * Adds all confusion matrix elements in another vector to this vector.
             *
             * @param begin A `const_iterator` to the beginning of the other vector
             * @param end   A `const_iterator` to the end of the other vector
             */
            void add(const_iterator begin, const_iterator end);

            /**
             * Adds the confusion matrix elements that correspond to an example at a specific index to this vector. The
             * confusion matrix elements to be added are multiplied by a specific weight.
             *
             * @param exampleIndex          The index of the example
             * @param labelMatrix           A reference to an object of type `CContiguousLabelMatrix` that provides
             *                              random access to the labels of the training examples
             * @param majorityLabelVector   A reference to an object of type `BinarySparseArrayVector` that stores the
             *                              predictions of the default rule
             * @param weightMatrix          A reference to an object of type `DenseWeightMatrix` that stores the weights
             *                              of individual examples and labels
             * @param weight                The weight, the confusion matrix elements should be multiplied by
             */
            void add(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                     const BinarySparseArrayVector& majorityLabelVector, const DenseWeightMatrix& weightMatrix,
                     float64 weight);

            /**
             * Adds the confusion matrix elements that correspond to an example at a specific index to this vector. The
             * confusion matrix elements to be added are multiplied by a specific weight.
             *
             * @param exampleIndex          The index of the example
             * @param labelMatrix           A reference to an object of type `CsrLabelMatrix` that provides row-wise
             *                              access to the labels of the training examples
             * @param majorityLabelVector   A reference to an object of type `BinarySparseArrayVector` that stores the
             *                              predictions of the default rule
             * @param weightMatrix          A reference to an object of type `DenseWeightMatrix` that stores the weights
             *                              of individual examples and labels
             * @param weight                The weight, the confusion matrix elements should be multiplied by
             */
            void add(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                     const BinarySparseArrayVector& majorityLabelVector, const DenseWeightMatrix& weightMatrix,
                     float64 weight);

            /**
             * Adds certain confusion matrix elements in another vector, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The confusion matrix elements to be added are multiplied by a
             * specific weight.
             *
             * @param exampleIndex          The index of the example
             * @param labelMatrix           A reference to an object of type `CContiguousLabelMatrix` that provides
             *                              random access to the labels of the training examples
             * @param majorityLabelVector   A reference to an object of type `BinarySparseArrayVector` that stores the
             *                              predictions of the default rule
             * @param weightMatrix          A reference to an object of type `DenseWeightMatrix` that stores the weights
             *                              of individual examples and labels
             * @param indices               A reference to a `CompleteIndexVector' that provides access to the indices
             * @param weight                The weight, the confusion matrix elements should be multiplied by
             */
            void addToSubset(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                             const BinarySparseArrayVector& majorityLabelVector, const DenseWeightMatrix& weightMatrix,
                             const CompleteIndexVector& indices, float64 weight);

            /**
             * Adds certain confusion matrix elements in another vector, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The confusion matrix elements to be added are multiplied by a
             * specific weight.
             *
             * @param exampleIndex          The index of the example
             * @param labelMatrix           A reference to an object of type `CsrLabelMatrix` that provides row-wise
             *                              access to the labels of the training examples
             * @param majorityLabelVector   A reference to an object of type `BinarySparseArrayVector` that stores the
             *                              predictions of the default rule
             * @param weightMatrix          A reference to an object of type `DenseWeightMatrix` that stores the weights
             *                              of individual examples and labels
             * @param indices               A reference to a `CompleteIndexVector' that provides access to the indices
             * @param weight                The weight, the confusion matrix elements should be multiplied by
             */
            void addToSubset(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                             const BinarySparseArrayVector& majorityLabelVector, const DenseWeightMatrix& weightMatrix,
                             const CompleteIndexVector& indices, float64 weight);

            /**
             * Adds certain confusion matrix elements in another vector, whose positions are given as a
             * `CompleteIndexVector`, to this vector. The confusion matrix elements to be added are multiplied by a
             * specific weight.
             *
             * @param exampleIndex          The index of the example
             * @param labelMatrix           A reference to an object of type `CContiguousLabelMatrix` that provides
             *                              random access to the labels of the training examples
             * @param majorityLabelVector   A reference to an object of type `BinarySparseArrayVector` that stores the
             *                              predictions of the default rule
             * @param weightMatrix          A reference to an object of type `DenseWeightMatrix` that stores the weights
             *                              of individual examples and labels
             * @param indices               A reference to a `PartialIndexVector' that provides access to the indices
             * @param weight                The weight, the confusion matrix elements should be multiplied by
             */
            void addToSubset(uint32 exampleIndex, const CContiguousLabelMatrix& labelMatrix,
                             const BinarySparseArrayVector& majorityLabelVector, const DenseWeightMatrix& weightMatrix,
                             const PartialIndexVector& indices, float64 weight);

            /**
             * Adds certain confusion matrix elements in another vector, whose positions are given as a
             * `PartialIndexVector`, to this vector. The confusion matrix elements to be added are multiplied by a
             * specific weight.
             *
             * @param exampleIndex          The index of the example
             * @param labelMatrix           A reference to an object of type `CsrLabelMatrix` that provides row-wise
             *                              access to the labels of the training examples
             * @param majorityLabelVector   A reference to an object of type `BinarySparseArrayVector` that stores the
             *                              predictions of the default rule
             * @param weightMatrix          A reference to an object of type `DenseWeightMatrix` that stores the weights
             *                              of individual examples and labels
             * @param indices               A reference to a `PartialIndexVector' that provides access to the indices
             * @param weight                The weight, the confusion matrix elements should be multiplied by
             */
            void addToSubset(uint32 exampleIndex, const CsrLabelMatrix& labelMatrix,
                             const BinarySparseArrayVector& majorityLabelVector, const DenseWeightMatrix& weightMatrix,
                             const PartialIndexVector& indices, float64 weight);

            /**
             * Sets the confusion matrix elements in this vector to the difference `first - second` between the elements
             * in two other vectors, considering only the elements in the first vector that correspond to the positions
             * provided by a `CompleteIndexVector`.
             *
             * @param firstBegin    A `const_iterator` to the beginning of the first vector
             * @param firstEnd      A `const_iterator` to the end of the first vector
             * @param firstIndices  A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices
             * @param secondBegin  A `const_iterator` to the beginning of the second vector
             * @param secondEnd    A `const_iterator` to the end of the second vector
             */
            void difference(const_iterator firstBegin, const_iterator firstEnd, const CompleteIndexVector& firstIndices,
                            const_iterator secondBegin, const_iterator secondEnd);

            /**
             * Sets the confusion matrix elements in this vector to the difference `first - second` between the elements
             * in two other vectors, considering only the elements in the first vector that correspond to the positions
             * provided by a `PartialIndexVector`.
             *
             * @param firstBegin    A `const_iterator` to the beginning of the first vector
             * @param firstEnd      A `const_iterator` to the end of the first vector
             * @param firstIndices  A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices
             * @param secondBegin   A `const_iterator` to the beginning of the second vector
             * @param secondEnd     A `const_iterator` to the end of the second vector
             */
            void difference(const_iterator firstBegin, const_iterator firstEnd, const PartialIndexVector& firstIndices,
                            const_iterator secondBegin, const_iterator secondEnd);

    };

}
