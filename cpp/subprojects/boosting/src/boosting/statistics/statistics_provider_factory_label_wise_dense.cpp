#include "boosting/statistics/statistics_provider_factory_label_wise_dense.hpp"
#include "common/validation.hpp"
#include "statistics_label_wise_dense.hpp"
#include "statistics_provider_label_wise.hpp"
#include "omp.h"


namespace boosting {

    template<typename LabelMatrix>
    static inline std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> createStatistics(
            const ILabelWiseLoss& lossFunction, const IEvaluationMeasure& evaluationMeasure,
            const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory, uint32 numThreads,
            const LabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<DenseLabelWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<DenseLabelWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<NumericDenseMatrix<float64>> scoreMatrixPtr =
            std::make_unique<NumericDenseMatrix<float64>>(numExamples, numLabels, true);
        const ILabelWiseLoss* lossFunctionRawPtr = &lossFunction;
        const LabelMatrix* labelMatrixPtr = &labelMatrix;
        const CContiguousConstView<float64>* scoreMatrixRawPtr = scoreMatrixPtr.get();
        DenseLabelWiseStatisticMatrix* statisticMatrixRawPtr = statisticMatrixPtr.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(lossFunctionRawPtr) \
        firstprivate(labelMatrixPtr) firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) \
        schedule(dynamic) num_threads(numThreads)
        for (uint32 i = 0; i < numExamples; i++) {
            lossFunctionRawPtr->updateLabelWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                          IndexIterator(labelMatrixPtr->getNumCols()),
                                                          *statisticMatrixRawPtr);
        }

        return std::make_unique<DenseLabelWiseStatistics<LabelMatrix>>(lossFunction, evaluationMeasure,
                                                                       ruleEvaluationFactory, labelMatrix,
                                                                       std::move(statisticMatrixPtr),
                                                                       std::move(scoreMatrixPtr));
    }

    DenseLabelWiseStatisticsProviderFactory::DenseLabelWiseStatisticsProviderFactory(
            std::unique_ptr<ILabelWiseLoss> lossFunctionPtr, std::unique_ptr<IEvaluationMeasure> evaluationMeasurePtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFunctionPtr_(std::move(lossFunctionPtr)), evaluationMeasurePtr_(std::move(evaluationMeasurePtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {
        assertNotNull("lossFunctionPtr", lossFunctionPtr_.get());
        assertNotNull("evaluationMeasurePtr", evaluationMeasurePtr_.get());
        assertNotNull("defaultRuleEvaluationFactoryPtr", defaultRuleEvaluationFactoryPtr_.get());
        assertNotNull("regularRuleEvaluationFactoryPtr", regularRuleEvaluationFactoryPtr_.get());
        assertNotNull("pruningRuleEvaluationFactoryPtr", pruningRuleEvaluationFactoryPtr_.get());
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    std::unique_ptr<IStatisticsProvider> DenseLabelWiseStatisticsProviderFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> statisticsPtr =
            createStatistics(*lossFunctionPtr_, *evaluationMeasurePtr_, *defaultRuleEvaluationFactoryPtr_, numThreads_,
                             labelMatrix);
        return std::make_unique<LabelWiseStatisticsProvider<ILabelWiseRuleEvaluationFactory>>(
            *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> DenseLabelWiseStatisticsProviderFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics<ILabelWiseRuleEvaluationFactory>> statisticsPtr =
            createStatistics(*lossFunctionPtr_, *evaluationMeasurePtr_, *defaultRuleEvaluationFactoryPtr_, numThreads_,
                             labelMatrix);
        return std::make_unique<LabelWiseStatisticsProvider<ILabelWiseRuleEvaluationFactory>>(
            *regularRuleEvaluationFactoryPtr_, *pruningRuleEvaluationFactoryPtr_, std::move(statisticsPtr));
    }

}
