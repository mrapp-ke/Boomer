/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"
#include "common/data/view_csr_binary.hpp"
#include <memory>


/**
 * Defines an interface for all measures that may be used to assess the quality of predictions for certain examples by
 * comparing them to the corresponding ground truth labels.
 */
class IEvaluationMeasure {

    public:

        virtual ~IEvaluationMeasure() { };

        /**
         * Calculates and returns a numerical score that assesses the quality of predictions for the example at a
         * specific index by comparing them to the corresponding ground truth labels, based on a label matrix that
         * provides random access to the labels of the training examples.
         *
         * @param exampleIndex  The index of the example for which the predictions should be evaluated
         * @param labelMatrix   A reference to an object of type `CContiguousConstView` that provides random access to
         *                      the labels of the training examples
         * @param scoreMatrix   A reference to an object of type `CContiguousConstView` that stores the currently 
         *                      predicted scores
         * @return              The numerical score that has been calculated
         */
        virtual float64 evaluate(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                 const CContiguousConstView<float64>& scoreMatrix) const = 0;

        /**
         * Calculates and returns a numerical score that assesses the quality of predictions for the example at a
         * specific index by comparing them to the corresponding ground truth labels, based on a label matrix that
         * provides row-wise access to the labels of the training examples.
         *
         * @param exampleIndex  The index of the example for which the predictions should be evaluated
         * @param labelMatrix   A reference to an object of type `BinaryCsrConstView` that provides row-wise access to
         *                      the labels of the training examples
         * @param scoreMatrix   A reference to an object of type `CContiguousConstView` that stores the currently 
         *                      predicted scores
         * @return              The numerical score that has been calculated
         */
        virtual float64 evaluate(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                 const CContiguousConstView<float64>& scoreMatrix) const = 0;

};

/**
 * Defines an interface for all factories that allow to create instances of the type `IEvaluationMeasure`.
 */
class IEvaluationMeasureFactory {

    public:

        virtual ~IEvaluationMeasureFactory() { };

        /**
         * Creates and returns a new object of type `IEvaluationMeasure`.
         *
         * @return An unique pointer to an object of type `IEvaluationMeasure` that has been created
         */
        virtual std::unique_ptr<IEvaluationMeasure> createEvaluationMeasure() const = 0;

};
