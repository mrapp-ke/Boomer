#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "boosting/statistics/statistics_provider_example_wise_dense.hpp"

#include "boosting/data/matrix_c_contiguous_numeric.hpp"
#include "boosting/data/statistic_vector_example_wise_dense.hpp"
#include "boosting/data/statistic_view_example_wise_dense.hpp"
#include "boosting/math/math.hpp"
#include "omp.h"
#include "statistics_example_wise_common.hpp"
#include "statistics_label_wise_dense.hpp"
#include "statistics_provider_example_wise.hpp"

#include <cstdlib>

namespace boosting {

    /**
     * A matrix that stores gradients and Hessians that have been calculated using a non-decomposable loss function
     * using C-contiguous arrays.
     */
    class DenseExampleWiseStatisticMatrix final : public DenseExampleWiseStatisticView {
        public:

            /**
             * @param numRows       The number of rows in the matrix
             * @param numGradients  The number of gradients per row
             */
            DenseExampleWiseStatisticMatrix(uint32 numRows, uint32 numGradients)
                : DenseExampleWiseStatisticView(
                  numRows, numGradients, triangularNumber(numGradients),
                  (float64*) malloc(numRows * numGradients * sizeof(float64)),
                  (float64*) malloc(numRows * triangularNumber(numGradients) * sizeof(float64))) {}

            ~DenseExampleWiseStatisticMatrix() {
                free(gradients_);
                free(hessians_);
            }
    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a differentiable loss function
     * that is applied example-wise and are stored using dense data structures.
     *
     * @tparam LabelMatrix The type of the matrix that provides access to the labels of the training examples
     */
    template<typename LabelMatrix>
    class DenseExampleWiseStatistics final
        : public AbstractExampleWiseStatistics<LabelMatrix, DenseExampleWiseStatisticVector,
                                               DenseExampleWiseStatisticView, DenseExampleWiseStatisticMatrix,
                                               NumericCContiguousMatrix<float64>, IExampleWiseLoss, IEvaluationMeasure,
                                               IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory> {
        public:

            /**
             * @param lossPtr               An unique pointer to an object of type `IExampleWiseLoss` that implements
             *                              the loss function to be used for calculating gradients and Hessians
             * @param evaluationMeasurePtr  An unique pointer to an object of type `IEvaluationMeasure` that implements
             *                              the evaluation measure that should be used to assess the quality of
             *                              predictions
             * @param ruleEvaluationFactory A reference to an object of type `IExampleWiseRuleEvaluationFactory`, to be
             *                              used for calculating the predictions, as well as corresponding quality
             *                              scores, of rules
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param statisticViewPtr      An unique pointer to an object of type `DenseExampleWiseStatisticView` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of type `NumericCContiguousMatrix` that
             *                              stores the currently predicted scores
             */
            DenseExampleWiseStatistics(std::unique_ptr<IExampleWiseLoss> lossPtr,
                                       std::unique_ptr<IEvaluationMeasure> evaluationMeasurePtr,
                                       const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                       const LabelMatrix& labelMatrix,
                                       std::unique_ptr<DenseExampleWiseStatisticView> statisticViewPtr,
                                       std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr)
                : AbstractExampleWiseStatistics<LabelMatrix, DenseExampleWiseStatisticVector,
                                                DenseExampleWiseStatisticView, DenseExampleWiseStatisticMatrix,
                                                NumericCContiguousMatrix<float64>, IExampleWiseLoss, IEvaluationMeasure,
                                                IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>(
                  std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, labelMatrix,
                  std::move(statisticViewPtr), std::move(scoreMatrixPtr)) {}

            /**
             * @see `IBoostingStatistics::visitScoreMatrix`
             */
            void visitScoreMatrix(IBoostingStatistics::DenseScoreMatrixVisitor denseVisitor,
                                  IBoostingStatistics::SparseScoreMatrixVisitor sparseVisitor) const override {
                denseVisitor(*this->scoreMatrixPtr_);
            }

            /**
             * @see `IExampleWiseStatistics::toLabelWiseStatistics`
             */
            std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> toLabelWiseStatistics(
              const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads) override final {
                uint32 numRows = this->statisticViewPtr_->getNumRows();
                uint32 numCols = this->statisticViewPtr_->getNumCols();
                std::unique_ptr<DenseLabelWiseStatisticView> labelWiseStatisticMatrixPtr =
                  std::make_unique<DenseLabelWiseStatisticMatrix>(numRows, numCols);
                DenseLabelWiseStatisticView* labelWiseStatisticMatrixRawPtr = labelWiseStatisticMatrixPtr.get();
                DenseExampleWiseStatisticView* exampleWiseStatisticViewRawPtr = this->statisticViewPtr_.get();

#pragma omp parallel for firstprivate(numRows) firstprivate(numCols) firstprivate(labelWiseStatisticMatrixRawPtr) \
  firstprivate(exampleWiseStatisticViewRawPtr) schedule(dynamic) num_threads(numThreads)
                for (int64 i = 0; i < numRows; i++) {
                    DenseLabelWiseStatisticView::iterator iterator = labelWiseStatisticMatrixRawPtr->begin(i);
                    DenseExampleWiseStatisticView::gradient_const_iterator gradientIterator =
                      exampleWiseStatisticViewRawPtr->gradients_cbegin(i);
                    DenseExampleWiseStatisticView::hessian_diagonal_const_iterator hessianIterator =
                      exampleWiseStatisticViewRawPtr->hessians_diagonal_cbegin(i);

                    for (uint32 j = 0; j < numCols; j++) {
                        Tuple<float64>& tuple = iterator[j];
                        tuple.first = gradientIterator[j];
                        tuple.second = hessianIterator[j];
                    }
                }

                return std::make_unique<DenseLabelWiseStatistics<LabelMatrix>>(
                  std::move(this->lossPtr_), std::move(this->evaluationMeasurePtr_), ruleEvaluationFactory,
                  this->labelMatrix_, std::move(labelWiseStatisticMatrixPtr), std::move(this->scoreMatrixPtr_));
            }
    };

    template<typename LabelMatrix>
    static inline std::unique_ptr<
      IExampleWiseStatistics<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>
      createStatistics(const IExampleWiseLossFactory& lossFactory,
                       const IEvaluationMeasureFactory& evaluationMeasureFactory,
                       const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads,
                       const LabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<IExampleWiseLoss> lossPtr = lossFactory.createExampleWiseLoss();
        std::unique_ptr<IEvaluationMeasure> evaluationMeasurePtr = evaluationMeasureFactory.createEvaluationMeasure();
        std::unique_ptr<DenseExampleWiseStatisticMatrix> statisticMatrixPtr =
          std::make_unique<DenseExampleWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr =
          std::make_unique<NumericCContiguousMatrix<float64>>(numExamples, numLabels, true);
        const IExampleWiseLoss* lossRawPtr = lossPtr.get();
        const LabelMatrix* labelMatrixPtr = &labelMatrix;
        const CContiguousConstView<float64>* scoreMatrixRawPtr = scoreMatrixPtr.get();
        DenseExampleWiseStatisticMatrix* statisticMatrixRawPtr = statisticMatrixPtr.get();

#pragma omp parallel for firstprivate(numExamples) firstprivate(lossRawPtr) firstprivate(labelMatrixPtr) \
  firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) num_threads(numThreads)
        for (int64 i = 0; i < numExamples; i++) {
            lossRawPtr->updateExampleWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr, *statisticMatrixRawPtr);
        }

        return std::make_unique<DenseExampleWiseStatistics<LabelMatrix>>(
          std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, labelMatrix,
          std::move(statisticMatrixPtr), std::move(scoreMatrixPtr));
    }

    DenseExampleWiseStatisticsProviderFactory::DenseExampleWiseStatisticsProviderFactory(
      std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr,
      std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
      std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<IExampleWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<IExampleWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {}

    std::unique_ptr<IStatisticsProvider> DenseExampleWiseStatisticsProviderFactory::create(
      const CContiguousConstView<const uint8>& labelMatrix) const {
        std::unique_ptr<IExampleWiseStatistics<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>
          statisticsPtr = createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_,
                                           *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<
          ExampleWiseStatisticsProvider<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> DenseExampleWiseStatisticsProviderFactory::create(
      const BinaryCsrConstView& labelMatrix) const {
        std::unique_ptr<IExampleWiseStatistics<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>
          statisticsPtr = createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_,
                                           *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<
          ExampleWiseStatisticsProvider<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    DenseConvertibleExampleWiseStatisticsProviderFactory::DenseConvertibleExampleWiseStatisticsProviderFactory(
      std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr,
      std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
      std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {}

    std::unique_ptr<IStatisticsProvider> DenseConvertibleExampleWiseStatisticsProviderFactory::create(
      const CContiguousConstView<const uint8>& labelMatrix) const {
        std::unique_ptr<IExampleWiseStatistics<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>
          statisticsPtr = createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_,
                                           *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<
          ConvertibleExampleWiseStatisticsProvider<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr), numThreads_);
    }

    std::unique_ptr<IStatisticsProvider> DenseConvertibleExampleWiseStatisticsProviderFactory::create(
      const BinaryCsrConstView& labelMatrix) const {
        std::unique_ptr<IExampleWiseStatistics<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>
          statisticsPtr = createStatistics(*lossFactoryPtr_, *evaluationMeasureFactoryPtr_,
                                           *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<
          ConvertibleExampleWiseStatisticsProvider<IExampleWiseRuleEvaluationFactory, ILabelWiseRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr), numThreads_);
    }

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
