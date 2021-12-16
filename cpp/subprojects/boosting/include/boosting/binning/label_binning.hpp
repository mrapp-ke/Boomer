/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"
#include <functional>
#include <memory>


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
     */
    class ILabelBinning {

        public:

            virtual ~ILabelBinning() { };

            /**
             * A callback function that is invoked when a label is assigned to a bin. It takes the index of the bin and
             * the index of the label as arguments.
             */
            typedef std::function<void(uint32 binIndex, uint32 labelIndex)> Callback;

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
             * Retrieves and returns information that is required to apply the binning method.
             *
             * This function must be called prior to the function `createBins` to obtain information, e.g. the number of
             * bins to be used, that is required to apply the binning method. This function may also be used to prepare,
             * e.g. sort, the given vector. The `LabelInfo` returned by this function must be passed to the function
             * `createBins` later on.
             *
             * @param criteria                  An array of type `float64` that stores the label-wise criteria that
             *                                  should be used to assign individual labels to bins
             * @param numElements               The number of elements in the array `criteria`
             * @return                          A struct of `type `LabelInfo` that stores the information
             */
            virtual LabelInfo getLabelInfo(const float64* criteria, uint32 numElements) const = 0;

            /**
             * Assigns the labels to bins based on label-wise criteria.
             *
             * @param labelInfo                 A struct of type `LabelInfo` that stores information that is required to
             *                                  apply the binning method
             * @param criteria                  An array of type `float64` that stores the label-wise criteria that
             *                                  should be used to assign individual labels to bins
             * @param numElements               The number of elements in the array `criteria`
             * @param callback                  A callback that is invoked when a label is assigned to a bin
             * @param zeroCallback              A callback that is invoked when a label for which the criterion is zero
             *                                  is encountered
             */
            virtual void createBins(LabelInfo labelInfo, const float64* criteria, uint32 numElements, Callback callback,
                                    ZeroCallback zeroCallback) const = 0;

    };

    /**
     * Defines an interface for all factories that allows to create instances of the type `ILabelBinning`.
     */
    class ILabelBinningFactory {

        public:

            virtual ~ILabelBinningFactory() { };

            /**
             * Creates and returns a new object of type `ILabelBinning`.
             *
             * @return An unique pointer to an object of type `ILabelBinning` that has been created
             */
            virtual std::unique_ptr<ILabelBinning> create() const = 0;

    };

}
