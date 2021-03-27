/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_evaluation/score_vector.hpp"
#include "common/data/vector_dense.hpp"


/**
 * An one-dimensional vector that stores the scores that may be predicted by a rule, as well as an overall quality
 * score that assesses the overall quality of the rule, in a C-contiguous array.
 *
 * @tparam T The type of the vector that provides access to the indices of the labels for which the rule may predict
 */
template<class T>
class DenseScoreVector : virtual public IScoreVector {

    private:

        const T& labelIndices_;

        DenseVector<float64> predictedScoreVector_;

    public:

        /**
         * @param labelIndices A reference to an object of template type `T` that provides access to the indices of the
         *                     labels for which the rule may predict
         */
        DenseScoreVector(const T& labelIndices);

        /**
         * An iterator that provides read-only access to the indices.
         */
        typedef typename T::const_iterator index_const_iterator;

        /**
         * An iterator that provides access to the predicted scores and allows to modify them.
         */
        typedef DenseVector<float64>::iterator score_iterator;

        /**
         * An iterator that provides read-only access to the predicted scores.
         */
        typedef DenseVector<float64>::const_iterator score_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        /**
         * Returns a `score_iterator` to the beginning of the predicted scores.
         *
         * @return A `score_iterator` to the beginning
         */
        score_iterator scores_begin();

        /**
         * Returns a `score_iterator` to the end of the predicted scores.
         *
         * @return A `score_iterator` to the end
         */
        score_iterator scores_end();

        /**
         * Returns a `score_const_iterator` to the beginning of the predicted scores.
         *
         * @return A `score_const_iterator` to the beginning
         */
        score_const_iterator scores_cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the predicted scores.
         *
         * @return A `const_iterator` to the end
         */
        score_const_iterator scores_cend() const;

        /**
         * Returns the number of labels for which the rule may predict.
         *
         * @return The number of labels
         */
        uint32 getNumElements() const;

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
