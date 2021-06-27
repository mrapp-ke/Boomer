/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <functional>


namespace boosting {

    /**
     * Stores information about a vector that provides access to the statistics for individual labels. This includes the
     * number of positive and negative bins, the labels should be assigned to, as well as the minimum and maximum
     * statistic in the vector.
     */
    struct LabelInfo {

        /**
         * The number of positive bins.
         */
        uint32 numPositiveBins;

        /**
         * The minimum among all statistics that belong to the positive bins.
         */
        float64 minPositive;

        /**
         * The maximum among all statistics that belong to the positive bins.
         */
        float64 maxPositive;

        /**
         * The number of negative bins.
         */
        uint32 numNegativeBins;

        /**
         * The minimum among all statistics that belong to the negative bins.
         */
        float64 minNegative;

        /**
         * The maximum among all statistics that belong to the negative bins.
         */
        float64 maxNegative;

    };

    /**
     * Defines an interface for methods that assign labels to bins, based on the corresponding gradients and Hessians.
     *
     * @tparam GradientIterator The type of the iterator that provides access to the gradients
     * @tparam HessianIterator  The type of the iterator that provides access to the Hessians
     */
    template<class GradientIterator, class HessianIterator>
    class ILabelBinning {

        public:

            virtual ~ILabelBinning() { };

            /**
             * A callback function that is invoked when a label is assigned to a bin. It takes the index of the bin, the
             * index of the label, as well as the statistic, as arguments.
             */
            typedef std::function<void(uint32 binIndex, uint32 labelIndex, float64 statistic)> Callback;

            /**
             * A callback function that is invoked when a label with zero statistics is encountered. It takes the index
             * of the label as an argument.
             */
            typedef std::function<void(uint32 labelIndex)> ZeroCallback;

            /**
             * Returns an upper bound for the number of bins used by the binning method, given a specific number of
             * labels for which rules may predict.
             *
             * @param numLabels The number of labels for which rules may predict
             * @return          The maximum number of bins used by the binning method
             */
            virtual uint32 getMaxBins(uint32 numLabels) const = 0;

            /**
             * Retrieves and returns information about the statistics for individual labels in a given vector that is
             * required to apply the binning method.
             *
             * This function must be called prior to the function `createBins` to obtain information, e.g. the number of
             * bins to be used, that is required to apply the binning method. This function may also be used to prepare,
             * e.g. sort, the given vector. The `LabelInfo` returned by this function must be passed to the function
             * `createBins` later on.
             *
             * @param gradientsBegin            An iterator to the beginning of the gradients
             * @param gradientsEnd              An iterator to the end of the gradients
             * @param hessiansBegin             An iterator to the beginning of the Hessians
             * @param hessiansEnd               An iterator to the end of the Hessians
             * @param l2RegularizationWeight    The weight to be used for L2 regularization
             * @return                          A struct of `type `LabelInfo` that stores the information
             */
            virtual LabelInfo getLabelInfo(GradientIterator gradientsBegin, GradientIterator gradientsEnd,
                                           HessianIterator hessiansBegin, HessianIterator hessiansEnd,
                                           float64 l2RegularizationWeight) const = 0;

            /**
             * Assigns the labels to bins, based on the corresponding statistics.
             *
             * @param labelInfo                 A struct of type `LabelInfo` that stores information about the
             *                                  statistics in the given vector
             * @param gradientsBegin            An iterator to the beginning of the gradients
             * @param gradientsEnd              An iterator to the end of the gradients
             * @param hessiansBegin             An iterator to the beginning of the Hessians
             * @param hessiansEnd               An iterator to the end of the Hessians
             * @param l2RegularizationWeight    The weight to be used for L2 regularization
             * @param callback                  A callback that is invoked when a label is assigned to a bin
             * @param zeroCallback              A callback that is invoked when a label with zero statistics is
             *                                  encountered
             */
            virtual void createBins(LabelInfo labelInfo, GradientIterator gradientsBegin, GradientIterator gradientsEnd,
                                    HessianIterator hessiansBegin, HessianIterator hessiansEnd,
                                    float64 l2RegularizationWeight, Callback callback,
                                    ZeroCallback zeroCallback) const = 0;

    };

}
