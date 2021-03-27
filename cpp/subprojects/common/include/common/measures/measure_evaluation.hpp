/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"
#include "common/input/label_matrix.hpp"


/**
 * Defines an interface for all measures that may be used to assess the quality of predictions for certain examples by
 * comparing them to the corresponding ground truth labels.
 */
class IEvaluationMeasure {

    public:

        virtual ~IEvaluationMeasure() { };

        /**
         * Calculates and returns a numerical score that assesses the quality of predictions for the example at a
         * specific index by comparing them to the corresponding ground truth labels.
         *
         * @param exampleIndex  The index of the example for which the predictions should be evaluated
         * @param labelMatrix   A reference to an object of type `IRandomAccessMatrix` that provides random access to
         *                      the labels of the training examples
         * @param scoreMatrix   A reference to an object of type `CContiguousView` that stores the currently predicted
         *                      scores
         * @return              The numerical score that has been calculated
         */
        virtual float64 evaluate(uint32 exampleIndex, const IRandomAccessLabelMatrix& labelMatrix,
                                 const CContiguousView<float64>& scoreMatrix) const = 0;

};
