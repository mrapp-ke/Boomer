/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"
#include "common/data/view_vector.hpp"
#include <memory>


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
         * @param relevantLabelIndices  A reference to an object of type `VectorConstView` that provides access to the
         *                              indices of the labels that are relevant to the given example
         * @param scoresBegin           An iterator to the beginning of the predicted scores
         * @param scoresEnd             An iterator to the end of the predicted scores
         * @return                      The numerical score that has been calculated
         */
        virtual float64 measureSimilarity(const VectorConstView<uint32>& relevantLabelIndices,
                                          CContiguousConstView<float64>::value_const_iterator scoresBegin,
                                          CContiguousConstView<float64>::value_const_iterator scoresEnd) const = 0;

};

/**
 * Defines an interface for all factories that allow to create instances of the type `ISimilarityMeasure`.
 */
class ISimilarityMeasureFactory {

    public:

        virtual ~ISimilarityMeasureFactory() { };

        /**
         * Creates and returns a new object of type `ISimilarityMeasure`.
         *
         * @return An unique pointer to an object of type `ISimilarityMeasure` that has been created
         */
        virtual std::unique_ptr<ISimilarityMeasure> createSimilarityMeasure() const = 0;

};
