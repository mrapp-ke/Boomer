#include "boosting/statistics/statistics_example_wise_dense.hpp"
#include "boosting/data/matrix_dense_numeric.hpp"
#include "boosting/data/matrix_dense_example_wise.hpp"
#include "boosting/data/vector_dense_example_wise.hpp"
#include "statistics_example_wise_common.hpp"
#include "omp.h"


namespace boosting {

    DenseExampleWiseStatisticsFactory::DenseExampleWiseStatisticsFactory(
            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr, uint32 numThreads)
        : lossFunctionPtr_(lossFunctionPtr), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr),
          labelMatrixPtr_(labelMatrixPtr), numThreads_(numThreads) {

    }

    std::unique_ptr<IExampleWiseStatistics> DenseExampleWiseStatisticsFactory::create() const {
        uint32 numExamples = labelMatrixPtr_->getNumRows();
        uint32 numLabels = labelMatrixPtr_->getNumCols();
        std::unique_ptr<DenseExampleWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<DenseExampleWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<DenseNumericMatrix<float64>> scoreMatrixPtr =
            std::make_unique<DenseNumericMatrix<float64>>(numExamples, numLabels, true);
        const IExampleWiseLoss* lossFunctionPtr = lossFunctionPtr_.get();
        const IRandomAccessLabelMatrix* labelMatrixPtr = labelMatrixPtr_.get();
        const CContiguousView<float64>* scoreMatrixRawPtr = scoreMatrixPtr.get();
        DenseExampleWiseStatisticMatrix* statisticMatrixRawPtr = statisticMatrixPtr.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(lossFunctionPtr) firstprivate(labelMatrixPtr) \
        firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            lossFunctionPtr->updateExampleWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr,
                                                         *statisticMatrixRawPtr);
        }

        return std::make_unique<ExampleWiseStatistics<DenseExampleWiseStatisticVector, DenseExampleWiseStatisticMatrix, DenseNumericMatrix<float64>>>(
            lossFunctionPtr_, ruleEvaluationFactoryPtr_, labelMatrixPtr_, std::move(statisticMatrixPtr),
            std::move(scoreMatrixPtr));
    }

}
