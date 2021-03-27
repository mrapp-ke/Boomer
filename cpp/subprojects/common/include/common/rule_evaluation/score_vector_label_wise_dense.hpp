/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/rule_evaluation/score_vector_label_wise.hpp"
#include "common/rule_evaluation/score_vector_dense.hpp"


/**
 * An one-dimensional vector that stores the scores that may be predicted by a rule, as well as corresponding quality
 * scores that assess the quality of individual scores, in C-contiguous arrays.
 *
 * @tparam T The type of the vector that provides access to the indices of the labels for which the rule may predict
 */
template<class T>
class DenseLabelWiseScoreVector final : public DenseScoreVector<T>, virtual public ILabelWiseScoreVector {

    private:

        DenseVector<float64> qualityScoreVector_;

    public:

        /**
         * @param labelIndices A reference to an object of template type `T` that provides access to the indices of the
         *                     labels for which the rule may predict
         */
        DenseLabelWiseScoreVector(const T& labelIndices);

        /**
         * An iterator that provides access to the quality scores and allows to modify them.
         */
        typedef DenseVector<float64>::iterator quality_score_iterator;

        /**
         * An iterator that provides read-only access to the quality scores.
         */
        typedef DenseVector<float64>::const_iterator quality_score_const_iterator;

        /**
         * Returns a `quality_score_iterator` to the beginning of the quality scores.
         *
         * @return A `quality_score_iterator` to the beginning
         */
        quality_score_iterator quality_scores_begin();

        /**
         * Returns a `quality_score_iterator` to the end of the quality scores.
         *
         * @return A `quality_score_iterator` to the end
         */
        quality_score_iterator quality_scores_end();

        /**
         * Returns a `quality_score_const_iterator` to the beginning of the quality scores.
         *
         * @return A `quality_score_const_iterator` to the beginning
         */
        quality_score_const_iterator quality_scores_cbegin() const;

        /**
         * Returns a `quality_score_const_iterator` to the end of the quality scores.
         *
         * @return A `quality_score_const_iterator` to the end
         */
        quality_score_const_iterator quality_scores_cend() const;

        const AbstractEvaluatedPrediction* processScores(const AbstractEvaluatedPrediction* bestHead,
                                                         ILabelWiseScoreProcessor& scoreProcessor) const override;

};
