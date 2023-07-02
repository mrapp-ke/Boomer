#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4250)
#endif

#include "boosting/statistics/statistics_provider_label_wise_dense.hpp"

#include "omp.h"
#include "statistics_label_wise_dense.hpp"
#include "statistics_provider_label_wise.hpp"

namespace boosting {

    template<typename LabelMatrix>
    static inline std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> createStatistics(
      const ILabelWiseLossFactory& lossFactory, const IEvaluationMeasureFactory& evaluationMeasureFactory,
      const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads, const LabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<ILabelWiseLoss> lossPtr = lossFactory.createLabelWiseLoss();
        std::unique_ptr<IEvaluationMeasure> evaluationMeasurePtr = evaluationMeasureFactory.createEvaluationMeasure();
        std::unique_ptr<DenseLabelWiseStatisticMatrix> statisticMatrixPtr =
          std::make_unique<DenseLabelWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<NumericCContiguousMatrix<float64>> scoreMatrixPtr =
          std::make_unique<NumericCContiguousMatrix<float64>>(numExamples, numLabels, true);
        const ILabelWiseLoss* lossRawPtr = lossPtr.get();
        const LabelMatrix* labelMatrixPtr = &labelMatrix;
        const CContiguousConstView<float64>* scoreMatrixRawPtr = scoreMatrixPtr.get();
        DenseLabelWiseStatisticMatrix* statisticMatrixRawPtr = statisticMatrixPtr.get();

#pragma omp parallel for firstprivate(numExamples) firstprivate(lossRawPtr) firstprivate(labelMatrixPtr) \
  firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) num_threads(numThreads)
        for (int64 i = 0; i < numExamples; i++) {
            lossRawPtr->updateLabelWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                  IndexIterator(labelMatrixPtr->getNumCols()), *statisticMatrixRawPtr);
        }

        return std::make_unique<DenseLabelWiseStatistics<LabelMatrix>>(
          std::move(lossPtr), std::move(evaluationMeasurePtr), ruleEvaluationFactory, labelMatrix,
          std::move(statisticMatrixPtr), std::move(scoreMatrixPtr));
    }

    DenseLabelWiseStatisticsProviderFactory::DenseLabelWiseStatisticsProviderFactory(
      std::unique_ptr<ILabelWiseLossFactory> lossFactoryPtr,
      std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
      std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
      std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
      std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFactoryPtr_(std::move(lossFactoryPtr)),
          evaluationMeasureFactoryPtr_(std::move(evaluationMeasureFactoryPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {}

    std::unique_ptr<IStatisticsProvider> DenseLabelWiseStatisticsProviderFactory::create(
      const CContiguousConstView<const uint8>& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> statisticsPtr = createStatistics(
          *lossFactoryPtr_, *evaluationMeasureFactoryPtr_, *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<LabelWiseStatisticsProvider<ILabelWiseRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> DenseLabelWiseStatisticsProviderFactory::create(
      const BinaryCsrConstView& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> statisticsPtr = createStatistics(
          *lossFactoryPtr_, *evaluationMeasureFactoryPtr_, *defaultRuleEvaluationFactoryPtr_, numThreads_, labelMatrix);
        return std::make_unique<LabelWiseStatisticsProvider<ILabelWiseRuleEvaluationFactory>>(
          *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

}

#ifdef _WIN32
    #pragma warning(pop)
#endif
