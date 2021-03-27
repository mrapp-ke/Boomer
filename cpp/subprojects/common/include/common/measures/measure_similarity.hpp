/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"
#include "common/input/label_vector.hpp"


/**
 * Defines an interface for all measures that may be used to compare predictions for certain examples to the
 * corresponding ground truth labels in order to quantify their similarity.
 */
class ISimilarityMeasure {

    public:

        virtual ~ISimilarityMeasure() { };

        /**
         * Calculates and returns a numerical score that quantifies the similarity of predictions for a single example
         * and the corresponding ground truth labels.
         *
         * @param labelVector   A reference to an object of type `LabelVector` that provides access to the relevant
         *                      labels of the given example
         * @param scoresBegin   An iterator to the beginning of the predicted scores
         * @param scoresEnd     An iterator to the end of the predicted scores
         * @return              The numerical score that has been calculated
         */
        virtual float64 measureSimilarity(const LabelVector& labelVector,
                                          CContiguousView<float64>::const_iterator scoresBegin,
                                          CContiguousView<float64>::const_iterator scoresEnd) const = 0;

};
