/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_evaluation/score_vector.hpp"
#include "common/data/vector_binned_dense.hpp"


/**
 * An one dimensional vector that stores the scores that may be predicted by a rule, corresponding to bins for which the
 * same prediction is made, as as an overall quality score that assesses the overall quality of the rule, in a
 * C-contiguous array.
 *
 * @tparam T The type of the vector that provides access to the indices of the labels for which the rule may predict
 */
template<class T>
class DenseBinnedScoreVector : virtual public IScoreVector {

    private:

        const T& labelIndices_;

        DenseBinnedVector<float64> binnedVector_;

    public:

        /**
         * @param labelIndices  A reference to an object of template type `T` that provides access to the indices of
         *                      the labels for which the rule may predict
         * @param numBins       The number of bins
         */
        DenseBinnedScoreVector(const T& labelIndices, uint32 numBins);

        /**
         * An iterator that provides read-only access to the indices of the labels for which the rule predicts.
         */
        typedef typename T::const_iterator index_const_iterator;

        /**
         * An iterator that provides read-only access to the predicted scores that correspond to individual labels.
         */
        typedef DenseBinnedVector<float64>::const_iterator score_const_iterator;

        /**
         * An iterator that provides access to the indices that correspond to individual bins and allows to modify them.
         */
        typedef DenseBinnedVector<float64>::index_binned_iterator index_binned_iterator;

        /**
         * An iterator that provides read-only access to the indices that correspond to individual bins.
         */
        typedef DenseBinnedVector<float64>::index_binned_const_iterator index_binned_const_iterator;

        /**
         * An iterator that provides access to the predicted scores that correspond to individual bins and allows to
         * modify them.
         */
        typedef DenseBinnedVector<float64>::binned_iterator score_binned_iterator;

        /**
         * An iterator that provides read-only access to the predicted scores that correspond to individual bins.
         */
        typedef DenseBinnedVector<float64>::binned_const_iterator score_binned_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices that correspond to individual labels.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices that correspond to individual labels.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        /**
         * Returns a `score_const_iterator` to the beginning of the predicted scores that correspond to individual
         * labels.
         *
         * @return A `score_const_iterator` to the beginning
         */
        score_const_iterator scores_cbegin() const;

        /**
         * Returns a `score_const_iterator` to the end of the predicted scores that correspond to individual labels.
         *
         * @return A `score_const_iterator` to the end
         */
        score_const_iterator scores_cend() const;

        /**
         * Returns an `index_binned_iterator` to the beginning of the indices that correspond to individual bins.
         *
         * @return An `index_binned_iterator` to the beginning
         */
        index_binned_iterator indices_binned_begin();

        /**
         * Returns an `index_binned_iterator` to the end of the indices that correspond to individual bins.
         *
         * @return An `index_binned_iterator` to the end
         */
        index_binned_iterator indices_binned_end();

        /**
         * Returns an `index_binned_const_iterator` to the beginning of the indices that correspond to individual bins.
         *
         * @return An `index_binned_const_iterator` to the beginning
         */
        index_binned_const_iterator indices_binned_cbegin() const;

        /**
         * Returns an `index_binned_const_iterator` to the end of the indices that correspond to individual bins.
         *
         * @return An `index_binned_const_iterator` to the end
         */
        index_binned_const_iterator indices_binned_cend() const;

        /**
         * Returns a `score_binned_iterator` to the beginning of the predicted scores that correspond to individual
         * bins.
         *
         * @return A `score_binned_iterator` to the beginning
         */
        score_binned_iterator scores_binned_begin();

        /**
         * Returns a `score_binned_iterator` to the end of the predicted scores that correspond to individual bins.
         *
         * @return A `score_binned_iterator` to the end
         */
        score_binned_iterator scores_binned_end();

        /**
         * Returns a `score_binned_const_iterator` to the beginning of the predicted scores that correspond to
         * individual bins.
         *
         * @return A `score_binned_const_iterator` to the beginning
         */
        score_binned_const_iterator scores_binned_cbegin() const;

        /**
         * Returns a `score_binned_const_iterator` to the end of the predicted scores that correspond to individual
         * bins.
         *
         * @return A `score_binned_const_iterator` to the end
         */
        score_binned_const_iterator scores_binned_cend() const;

        /**
         * Returns the number of labels for which the rule may predict.
         *
         * @return The number of labels
         */
        uint32 getNumElements() const;

        /**
         * Returns the number of bins.
         *
         * @return The number of bins
         */
        uint32 getNumBins() const;

        /**
         * Sets the number of bins.
         *
         * @param numBins       The number of bins to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumBins(uint32 numBins, bool freeMemory);

        /**
         * Returns whether the rule may only predict for a subset of the available labels, or not.
         *
         * @return True, if the rule may only predict for a subset of the available labels, false otherwise
         */
        bool isPartial() const;

        void updatePrediction(AbstractPrediction& prediction) const override final;

        const AbstractEvaluatedPrediction* processScores(const AbstractEvaluatedPrediction* bestHead,
                                                         IScoreProcessor& scoreProcessor) const override final;

};
