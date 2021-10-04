/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/model/body.hpp"
#include "common/model/condition_list.hpp"


/**
 * A body that consists of a conjunction of conditions using the operators <= or > for numerical conditions, and = or !=
 * for nominal conditions, respectively.
 */
class ConjunctiveBody final : public IBody {

    private:

        uint32 numLeq_;

        uint32* leqFeatureIndices_;

        float32* leqThresholds_;

        uint32 numGr_;

        uint32* grFeatureIndices_;

        float32* grThresholds_;

        uint32 numEq_;

        uint32* eqFeatureIndices_;

        float32* eqThresholds_;

        uint32 numNeq_;

        uint32* neqFeatureIndices_;

        float32* neqThresholds_;

    public:

        /**
         * @param numLeq    The number of conditions that use the <= operator
         * @param numGr     The number of conditions that use the > operator
         * @param numEq     The number of conditions that use the == operator
         * @param numNeq    The number of conditions that use the != operator
         */
        ConjunctiveBody(uint32 numLeq, uint32 numGr, uint32 numEq, uint32 numNeq);

        /**
         * @param conditionList A reference to an object of type `ConditionList` that provides access to the conditions,
         *                      the body should contain
         */
        ConjunctiveBody(const ConditionList& conditionList);

        ~ConjunctiveBody();

        /**
         * An iterator that provides access to the thresholds that are used by the conditions in the body and allows to
         * modify them.
         */
        typedef float32* threshold_iterator;

        /**
         * An iterator that provides read-only access to the thresholds that are used by the conditions in the body.
         */
        typedef const float32* threshold_const_iterator;

        /**
         * An iterator that provides access to the feature indices that correspond to the conditions in the body and
         * allows to modify them.
         */
        typedef uint32* index_iterator;

        /**
         * An iterator that provides read-only access to the feature indices that correspond to the conditions in the
         * body.
         */
        typedef const uint32* index_const_iterator;

        /**
         * Returns the number of conditions that use the <= operator.
         *
         * @return The number of conditions
         */
        uint32 getNumLeq() const;

        /**
         * Returns a `threshold_iterator` to the beginning of the thresholds that correspond to conditions that use the
         * <= operator.
         *
         * @return A `threshold_iterator` to the beginning
         */
        threshold_iterator leq_thresholds_begin();

        /**
         * Returns a `threshold_iterator` to the end of the thresholds that correspond to conditions that use the <=
         * operator.
         *
         * @return A `threshold_iterator` to the end
         */
        threshold_iterator leq_thresholds_end();

        /**
         * Returns a `threshold_const_iterator` to the beginning of the thresholds that correspond to conditions that
         * use the <= operator.
         *
         * @return A `threshold_const_iterator` to the beginning
         */
        threshold_const_iterator leq_thresholds_cbegin() const;

        /**
         * Returns a `threshold_const_iterator` to the end of the thresholds that correspond to conditions that use the
         * <= operator.
         *
         * @return A `threshold_const_iterator` to the end
         */
        threshold_const_iterator leq_thresholds_cend() const;

        /**
         * Returns an `index_iterator` to the beginning of the feature indices that correspond to conditions that use the
         * <= operator.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator leq_indices_begin();

        /**
         * Returns an `index_iterator` to the end of the feature indices that correspond to conditions that use the <=
         * operator.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator leq_indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the feature indices that correspond to conditions that
         * use the <= operator.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator leq_indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the feature indices that correspond to conditions that use
         * the <= operator.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator leq_indices_cend() const;

        /**
         * Returns the number of conditions that use the > operator.
         *
         * @return The number of conditions
         */
        uint32 getNumGr() const;

        /**
         * Returns a `threshold_iterator` to the beginning of the thresholds that correspond to conditions that use the
         * > operator.
         *
         * @return A `threshold_iterator` to the beginning
         */
        threshold_iterator gr_thresholds_begin();

        /**
         * Returns a `threshold_iterator` to the end of the thresholds that correspond to conditions that use the >
         * operator.
         *
         * @return A `threshold_iterator` to the end
         */
        threshold_iterator gr_thresholds_end();

        /**
         * Returns a `threshold_const_iterator` to the beginning of the thresholds that correspond to conditions that
         * use the > operator.
         *
         * @return A `threshold_const_iterator` to the beginning
         */
        threshold_const_iterator gr_thresholds_cbegin() const;

        /**
         * Returns a `threshold_const_iterator` to the end of the thresholds that correspond to conditions that use the
         * > operator.
         *
         * @return A `threshold_const_iterator` to the end
         */
        threshold_const_iterator gr_thresholds_cend() const;

        /**
         * Returns an `index_iterator` to the beginning of the feature indices that correspond to conditions that use
         * the > operator.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator gr_indices_begin();

        /**
         * Returns an `index_iterator` to the end of the feature indices that correspond to conditions that use the >
         * operator.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator gr_indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the feature indices that correspond to conditions that
         * use the > operator.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator gr_indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the feature indices that correspond to conditions that use
         * the > operator.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator gr_indices_cend() const;

        /**
         * Returns the number of conditions that use the == operator.
         *
         * @return The number of conditions
         */
        uint32 getNumEq() const;

        /**
         * Returns a `threshold_iterator` to the beginning of the thresholds that correspond to conditions that use the
         * == operator.
         *
         * @return A `threshold_iterator` to the beginning
         */
        threshold_iterator eq_thresholds_begin();

        /**
         * Returns a `threshold_iterator` to the end of the thresholds that correspond to conditions that use the ==
         * operator.
         *
         * @return A `threshold_iterator` to the end
         */
        threshold_iterator eq_thresholds_end();

        /**
         * Returns a `threshold_const_iterator` to the beginning of the thresholds that correspond to conditions that
         * use the == operator.
         *
         * @return A `threshold_const_iterator` to the beginning
         */
        threshold_const_iterator eq_thresholds_cbegin() const;

        /**
         * Returns a `threshold_const_iterator` to the end of the thresholds that correspond to conditions that use the
         * == operator.
         *
         * @return A `threshold_const_iterator` to the end
         */
        threshold_const_iterator eq_thresholds_cend() const;

        /**
         * Returns an `index_iterator` to the beginning of the feature indices that correspond to conditions that use
         * the == operator.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator eq_indices_begin();

        /**
         * Returns an `index_iterator` to the end of the feature indices that correspond to conditions that use the ==
         * operator.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator eq_indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the feature indices that correspond to conditions that
         * use the == operator.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator eq_indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the feature indices that correspond to conditions that use
         * the == operator.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator eq_indices_cend() const;

        /**
         * Returns the number of conditions that use the != operator.
         *
         * @return The number of conditions
         */
        uint32 getNumNeq() const;

        /**
         * Returns a `threshold_iterator` to the beginning of the thresholds that correspond to conditions that use the
         * != operator.
         *
         * @return A `threshold_iterator` to the beginning
         */
        threshold_iterator neq_thresholds_begin();

        /**
         * Returns a `threshold_iterator` to the end of the thresholds that correspond to conditions that use the !=
         * operator.
         *
         * @return A `threshold_iterator` to the end
         */
        threshold_iterator neq_thresholds_end();

        /**
         * Returns a `threshold_const_iterator` to the beginning of the thresholds that correspond to conditions that
         * use the != operator.
         *
         * @return A `threshold_const_iterator` to the beginning
         */
        threshold_const_iterator neq_thresholds_cbegin() const;

        /**
         * Returns a `threshold_const_iterator` to the end of the thresholds that correspond to conditions that use the
         * != operator.
         *
         * @return A `threshold_const_iterator` to the end
         */
        threshold_const_iterator neq_thresholds_cend() const;

        /**
         * Returns an `index_iterator` to the beginning of the feature indices that correspond to conditions that use
         * the != operator.
         *
         * @return An `index_iterator` to the beginning
         */
        index_iterator neq_indices_begin();

        /**
         * Returns an `index_iterator` to the end of the feature indices that correspond to conditions that use the !=
         * operator.
         *
         * @return An `index_iterator` to the end
         */
        index_iterator neq_indices_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the feature indices that correspond to conditions that
         * use the != operator.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator neq_indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the feature indices that correspond to conditions that use
         * the != operator.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator neq_indices_cend() const;

        bool covers(CContiguousFeatureMatrix::const_iterator begin,
                    CContiguousFeatureMatrix::const_iterator end) const override;

        bool covers(CsrFeatureMatrix::index_const_iterator indicesBegin,
                    CsrFeatureMatrix::index_const_iterator indicesEnd,
                    CsrFeatureMatrix::value_const_iterator valuesBegin,
                    CsrFeatureMatrix::value_const_iterator valuesEnd, float32* tmpArray1, uint32* tmpArray2,
                    uint32 n) const override;

        void visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor) const override;

};
