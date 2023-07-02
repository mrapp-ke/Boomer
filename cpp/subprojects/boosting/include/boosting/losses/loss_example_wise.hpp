/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/data/statistic_view_example_wise_dense.hpp"
#include "boosting/losses/loss_label_wise.hpp"

namespace boosting {

    /**
     * Defines an interface for all (non-decomposable) loss functions that are applied example-wise.
     */
    class IExampleWiseLoss : public ILabelWiseLoss {
        public:

            virtual ~IExampleWiseLoss() override {};

            /**
             * Updates the statistics of the example at a specific index.
             *
             * @param exampleIndex  The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix   A reference to an object of type `CContiguousConstView` that provides random access
             *                      to the labels of the training examples
             * @param scoreMatrix   A reference to an object of type `CContiguousConstView` that stores the currently
             *                      predicted scores
             * @param statisticView A reference to an object of type `DenseExampleWiseStatisticView` to be updated
             */
            virtual void updateExampleWiseStatistics(uint32 exampleIndex,
                                                     const CContiguousConstView<const uint8>& labelMatrix,
                                                     const CContiguousConstView<float64>& scoreMatrix,
                                                     DenseExampleWiseStatisticView& statisticView) const = 0;

            /**
             * Updates the statistics of the example at a specific index.
             *
             * @param exampleIndex  The index of the example for which the gradients and Hessians should be updated
             * @param labelMatrix   A reference to an object of type `BinaryCsrConstView` that provides row-wise access
             *                      to the labels of the training examples
             * @param scoreMatrix   A reference to an object of type `CContiguousConstView` that stores the currently
             *                      predicted scores
             * @param statisticView A reference to an object of type `DenseExampleWiseStatisticView` to be updated
             */
            virtual void updateExampleWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                     const CContiguousConstView<float64>& scoreMatrix,
                                                     DenseExampleWiseStatisticView& statisticView) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `IExampleWiseLoss`.
     */
    class IExampleWiseLossFactory : public ILabelWiseLossFactory {
        public:

            virtual ~IExampleWiseLossFactory() override {};

            /**
             * Creates and returns a new object of type `IExampleWiseLoss`.
             *
             * @return An unique pointer to an object of type `IExampleWiseLoss` that has been created
             */
            virtual std::unique_ptr<IExampleWiseLoss> createExampleWiseLoss() const = 0;

            std::unique_ptr<ILabelWiseLoss> createLabelWiseLoss() const override final {
                return this->createExampleWiseLoss();
            }
    };

    /**
     * Defines an interface for all classes that allow to configure a (non-decomposable) loss function that is applied
     * example-wise.
     */
    class IExampleWiseLossConfig : public ILossConfig {
        public:

            virtual ~IExampleWiseLossConfig() override {};

            /**
             * Creates and returns a new object of type `IExampleWiseLossFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `IExampleWiseLossFactory` that has been created
             */
            virtual std::unique_ptr<IExampleWiseLossFactory> createExampleWiseLossFactory() const = 0;

            std::unique_ptr<IEvaluationMeasureFactory> createEvaluationMeasureFactory() const override final {
                return this->createExampleWiseLossFactory();
            }

            std::unique_ptr<IDistanceMeasureFactory> createDistanceMeasureFactory() const override final {
                return this->createExampleWiseLossFactory();
            }

            bool isDecomposable() const override final {
                return false;
            }

            bool isSparse() const override {
                return true;
            }
    };

}
