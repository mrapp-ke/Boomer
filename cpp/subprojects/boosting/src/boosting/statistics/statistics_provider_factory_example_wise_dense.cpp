#include "boosting/statistics/statistics_provider_factory_example_wise_dense.hpp"
#include "boosting/data/matrix_dense_numeric.hpp"
#include "boosting/data/statistic_vector_dense_example_wise.hpp"
#include "boosting/data/statistic_view_dense_example_wise.hpp"
#include "boosting/math/math.hpp"
#include "common/validation.hpp"
#include "statistics_example_wise_common.hpp"
#include "statistics_label_wise_dense.hpp"
#include "statistics_provider_example_wise.hpp"
#include "omp.h"
#include <cstdlib>


namespace boosting {

    /**
     * A matrix that stores gradients and Hessians that have been calculated using a non-decomposable loss function
     * using C-contiguous arrays.
     */
    class DenseExampleWiseStatisticMatrix : public DenseExampleWiseStatisticView {

        public:

            /**
             * @param numRows       The number of rows in the matrix
             * @param numGradients  The number of gradients per row
             */
            DenseExampleWiseStatisticMatrix(uint32 numRows, uint32 numGradients)
                : DenseExampleWiseStatisticView(
                      numRows, numGradients, triangularNumber(numGradients),
                      (float64*) malloc(numRows * numGradients * sizeof(float64)),
                      (float64*) malloc(numRows * triangularNumber(numGradients) * sizeof(float64))) {

            }

    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a differentiable loss function
     * that is applied example-wise and are stored using dense data structures.
     *
     * @tparam LabelMatrix The type of the matrix that provides access to the labels of the training examples
     */
    template<typename LabelMatrix>
    class DenseExampleWiseStatistics final : public AbstractExampleWiseStatistics<LabelMatrix,
                                                                                  DenseExampleWiseStatisticVector,
                                                                                  DenseExampleWiseStatisticView,
                                                                                  DenseExampleWiseStatisticMatrix,
                                                                                  NumericDenseMatrix<float64>,
                                                                                  IExampleWiseLoss> {

        public:

            /**
             * @param lossFunction          A reference to an object of type `IExampleWiseLoss`, representing the loss
             *                              function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of type `IExampleWiseRuleEvaluationFactory`, to be
             *                              used for calculating the predictions, as well as corresponding quality
             *                              scores, of rules
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param statisticViewPtr      An unique pointer to an object of type `DenseExampleWiseStatisticView` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of type `NumericDenseMatrix` that stores the
             *                              currently predicted scores
             */
            DenseExampleWiseStatistics(const IExampleWiseLoss& lossFunction,
                                       const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                       const LabelMatrix& labelMatrix,
                                       std::unique_ptr<DenseExampleWiseStatisticView> statisticViewPtr,
                                       std::unique_ptr<NumericDenseMatrix<float64>> scoreMatrixPtr)
                : AbstractExampleWiseStatistics<LabelMatrix, DenseExampleWiseStatisticVector,
                                                DenseExampleWiseStatisticView, DenseExampleWiseStatisticMatrix,
                                                NumericDenseMatrix<float64>, IExampleWiseLoss>(
                      lossFunction, ruleEvaluationFactory, labelMatrix, std::move(statisticViewPtr),
                      std::move(scoreMatrixPtr)) {

            }

            /**
             * @see `IExampleWiseStatistics::toLabelWiseStatistics`
             */
            std::unique_ptr<ILabelWiseStatistics> toLabelWiseStatistics(
                    const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads) override final {
                uint32 numRows = this->statisticViewPtr_->getNumRows();
                uint32 numCols = this->statisticViewPtr_->getNumCols();
                std::unique_ptr<DenseLabelWiseStatisticView> labelWiseStatisticMatrixPtr =
                    std::make_unique<DenseLabelWiseStatisticMatrix>(numRows, numCols);
                DenseLabelWiseStatisticView* labelWiseStatisticMatrixRawPtr = labelWiseStatisticMatrixPtr.get();
                DenseExampleWiseStatisticView* exampleWiseStatisticViewRawPtr = this->statisticViewPtr_.get();

                #pragma omp parallel for firstprivate(numRows) firstprivate(numCols) \
                firstprivate(labelWiseStatisticMatrixRawPtr) firstprivate(exampleWiseStatisticViewRawPtr) \
                schedule(dynamic) num_threads(numThreads)
                for (uint32 i = 0; i < numRows; i++) {
                    DenseLabelWiseStatisticView::iterator iterator = labelWiseStatisticMatrixRawPtr->row_begin(i);
                    DenseExampleWiseStatisticView::gradient_const_iterator gradientIterator =
                        exampleWiseStatisticViewRawPtr->gradients_row_cbegin(i);
                    DenseExampleWiseStatisticView::hessian_diagonal_const_iterator hessianIterator =
                        exampleWiseStatisticViewRawPtr->hessians_diagonal_row_cbegin(i);

                    for (uint32 j = 0; j < numCols; j++) {
                        Tuple<float64>& tuple = iterator[j];
                        tuple.first = gradientIterator[j];
                        tuple.second = hessianIterator[j];
                    }
                }

                return std::make_unique<DenseLabelWiseStatistics<LabelMatrix>>(this->lossFunction_,
                                                                               ruleEvaluationFactory,
                                                                               this->labelMatrix_,
                                                                               std::move(labelWiseStatisticMatrixPtr),
                                                                               std::move(this->scoreMatrixPtr_));
            }

    };

    template<typename LabelMatrix>
    static inline std::unique_ptr<IExampleWiseStatistics> createInternally(
            const IExampleWiseLoss& lossFunction, const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory,
            uint32 numThreads, const LabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<DenseExampleWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<DenseExampleWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<NumericDenseMatrix<float64>> scoreMatrixPtr =
            std::make_unique<NumericDenseMatrix<float64>>(numExamples, numLabels, true);
        const IExampleWiseLoss* lossFunctionRawPtr = &lossFunction;
        const LabelMatrix* labelMatrixPtr = &labelMatrix;
        const CContiguousConstView<float64>* scoreMatrixRawPtr = scoreMatrixPtr.get();
        DenseExampleWiseStatisticMatrix* statisticMatrixRawPtr = statisticMatrixPtr.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(lossFunctionRawPtr) \
        firstprivate(labelMatrixPtr) firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) \
        schedule(dynamic) num_threads(numThreads)
        for (uint32 i = 0; i < numExamples; i++) {
            lossFunctionRawPtr->updateExampleWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr,
                                                            *statisticMatrixRawPtr);
        }

        return std::make_unique<DenseExampleWiseStatistics<LabelMatrix>>(
            lossFunction, ruleEvaluationFactory, labelMatrix, std::move(statisticMatrixPtr),
            std::move(scoreMatrixPtr));
    }

    /**
     * A factory that allows to create new instances of the class `IExampleWiseStatistics` that used dense data
     * structures to store the statistics.
     */
    class DenseExampleWiseStatisticsFactory final : public IExampleWiseStatisticsFactory {

        private:

            const IExampleWiseLoss& lossFunction_;

            const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory_;

            uint32 numThreads_;

        public:

            /**
             * @param lossFunction          A reference to an object of type `IExampleWiseLoss`, representing the loss
             *                              function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of type `IExampleWiseRuleEvaluationFactory`, to be
             *                              used for calculating the predictions, as well as corresponding quality
             *                              scores, of rules
             * @param numThreads            The number of CPU threads to be used to calculate the initial statistics in
             *                              parallel. Must be at least 1
             */
            DenseExampleWiseStatisticsFactory(const IExampleWiseLoss& lossFunction,
                                              const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                              uint32 numThreads)
                : lossFunction_(lossFunction), ruleEvaluationFactory_(ruleEvaluationFactory), numThreads_(numThreads) {

            }

            std::unique_ptr<IExampleWiseStatistics> create(const CContiguousLabelMatrix& labelMatrix) const override {
                return createInternally<CContiguousLabelMatrix>(lossFunction_, ruleEvaluationFactory_, numThreads_,
                                                                labelMatrix);
            }

            std::unique_ptr<IExampleWiseStatistics> create(const CsrLabelMatrix& labelMatrix) const override {
                return createInternally<CsrLabelMatrix>(lossFunction_, ruleEvaluationFactory_, numThreads_,
                                                        labelMatrix);
            }

    };

    DenseExampleWiseStatisticsProviderFactory::DenseExampleWiseStatisticsProviderFactory(
            std::unique_ptr<IExampleWiseLoss> lossFunctionPtr,
            std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
            std::unique_ptr<IExampleWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
            std::unique_ptr<IExampleWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFunctionPtr_(std::move(lossFunctionPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {
        assertNotNull("lossFunctionPtr", lossFunctionPtr_.get());
        assertNotNull("defaultRuleEvaluationFactoryPtr", defaultRuleEvaluationFactoryPtr_.get());
        assertNotNull("regularRuleEvaluationFactoryPtr", regularRuleEvaluationFactoryPtr_.get());
        assertNotNull("pruningRuleEvaluationFactoryPtr", pruningRuleEvaluationFactoryPtr_.get());
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    std::unique_ptr<IStatisticsProvider> DenseExampleWiseStatisticsProviderFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        DenseExampleWiseStatisticsFactory statisticsFactory(*lossFunctionPtr_, *defaultRuleEvaluationFactoryPtr_,
                                                            numThreads_);
        return std::make_unique<ExampleWiseStatisticsProvider>(*regularRuleEvaluationFactoryPtr_,
                                                               *pruningRuleEvaluationFactoryPtr_,
                                                               statisticsFactory.create(labelMatrix));
    }

    std::unique_ptr<IStatisticsProvider> DenseExampleWiseStatisticsProviderFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        DenseExampleWiseStatisticsFactory statisticsFactory(*lossFunctionPtr_, *defaultRuleEvaluationFactoryPtr_,
                                                            numThreads_);
        return std::make_unique<ExampleWiseStatisticsProvider>(*regularRuleEvaluationFactoryPtr_,
                                                               *pruningRuleEvaluationFactoryPtr_,
                                                               statisticsFactory.create(labelMatrix));
    }

    DenseConvertibleExampleWiseStatisticsProviderFactory::DenseConvertibleExampleWiseStatisticsProviderFactory(
            std::unique_ptr<IExampleWiseLoss> lossFunctionPtr,
            std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFunctionPtr_(std::move(lossFunctionPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {
        assertNotNull("lossFunctionPtr", lossFunctionPtr_.get());
        assertNotNull("defaultRuleEvaluationFactoryPtr", defaultRuleEvaluationFactoryPtr_.get());
        assertNotNull("regularRuleEvaluationFactoryPtr", regularRuleEvaluationFactoryPtr_.get());
        assertNotNull("pruningRuleEvaluationFactoryPtr", pruningRuleEvaluationFactoryPtr_.get());
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    std::unique_ptr<IStatisticsProvider> DenseConvertibleExampleWiseStatisticsProviderFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        DenseExampleWiseStatisticsFactory statisticsFactory(*lossFunctionPtr_, *defaultRuleEvaluationFactoryPtr_,
                                                            numThreads_);
        return std::make_unique<ConvertibleExampleWiseStatisticsProvider>(*regularRuleEvaluationFactoryPtr_,
                                                                          *pruningRuleEvaluationFactoryPtr_,
                                                                          statisticsFactory.create(labelMatrix),
                                                                          numThreads_);
    }

    std::unique_ptr<IStatisticsProvider> DenseConvertibleExampleWiseStatisticsProviderFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        DenseExampleWiseStatisticsFactory statisticsFactory(*lossFunctionPtr_, *defaultRuleEvaluationFactoryPtr_,
                                                            numThreads_);
        return std::make_unique<ConvertibleExampleWiseStatisticsProvider>(*regularRuleEvaluationFactoryPtr_,
                                                                          *pruningRuleEvaluationFactoryPtr_,
                                                                          statisticsFactory.create(labelMatrix),
                                                                          numThreads_);
    }

}
