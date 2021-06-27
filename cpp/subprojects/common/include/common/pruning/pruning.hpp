/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/model/condition_list.hpp"
#include "common/sampling/partition.hpp"
#include "common/thresholds/thresholds_subset.hpp"


/**
 * Defines an interface for all classes that implement a strategy for pruning classification rules based on a
 * "prune set", i.e., based on the examples that are not contained in the sub-sample of the training data that has been
 * used to learn the rule, referred to a the "grow set".
 */
class IPruning {

    public:

        virtual ~IPruning() { };

        /**
         * Prunes the conditions of an existing rule by modifying a given list of conditions in-place. The rule is
         * pruned by removing individual conditions in a way that improves over its original quality score as measured
         * on the the prune set.
         *
         * @param thresholdsSubset  A reference to an object of type `IThresholdsSubset`, which contains the thresholds
         *                          that correspond to the subspace of the instance space that is covered by the
         *                          existing rule
         * @param partition         A reference to an object of type `IPartition` that provides access to the indices of
         *                          the training examples that belong to the training set and the holdout set,
         *                          respectively
         * @param conditions        A reference to an object of type `ConditionList` that stores the conditions of the
         *                          existing rule
         * @param head              A reference to an object of type `AbstractPrediction` that stores the scores that
         *                          are predicted by the existing rule
         * @return                  An unique pointer to an object of type `ICoverageState` that keeps track of the
         *                          examples that are covered by the pruned rule or a null pointer if the rule was not
         *                          pruned
         */
        virtual std::unique_ptr<ICoverageState> prune(IThresholdsSubset& thresholdsSubset, IPartition& partition,
                                                      ConditionList& conditions,
                                                      const AbstractPrediction& head) const = 0;

};
