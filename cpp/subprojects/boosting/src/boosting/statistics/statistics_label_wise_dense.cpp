#include "boosting/statistics/statistics_label_wise_dense.hpp"
#include "boosting/data/matrix_dense_numeric.hpp"
#include "boosting/data/matrix_dense_label_wise.hpp"
#include "boosting/data/vector_dense_label_wise.hpp"
#include "statistics_label_wise_common.hpp"
#include "omp.h"


namespace boosting {

    DenseLabelWiseStatisticsFactory::DenseLabelWiseStatisticsFactory(
            std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr, uint32 numThreads)
        : lossFunctionPtr_(lossFunctionPtr), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr),
          labelMatrixPtr_(labelMatrixPtr), numThreads_(numThreads) {

    }

    std::unique_ptr<ILabelWiseStatistics> DenseLabelWiseStatisticsFactory::create() const {
        uint32 numExamples = labelMatrixPtr_->getNumRows();
        uint32 numLabels = labelMatrixPtr_->getNumCols();
        std::unique_ptr<DenseLabelWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<DenseLabelWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<DenseNumericMatrix<float64>> scoreMatrixPtr =
            std::make_unique<DenseNumericMatrix<float64>>(numExamples, numLabels, true);
        FullIndexVector labelIndices(numLabels);
        const ILabelWiseLoss* lossFunctionPtr = lossFunctionPtr_.get();
        const IRandomAccessLabelMatrix* labelMatrixPtr = labelMatrixPtr_.get();
        const FullIndexVector* labelIndicesPtr = &labelIndices;
        const CContiguousView<float64>* scoreMatrixRawPtr = scoreMatrixPtr.get();
        DenseLabelWiseStatisticMatrix* statisticMatrixRawPtr = statisticMatrixPtr.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(lossFunctionPtr) firstprivate(labelMatrixPtr) \
        firstprivate(scoreMatrixRawPtr) firstprivate(labelIndicesPtr) firstprivate(statisticMatrixRawPtr) \
        schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            lossFunctionPtr->updateLabelWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr,
                                                       labelIndicesPtr->cbegin(), labelIndicesPtr->cend(),
                                                       *statisticMatrixRawPtr);
        }

        return std::make_unique<LabelWiseStatistics<DenseLabelWiseStatisticVector, DenseLabelWiseStatisticMatrix, DenseNumericMatrix<float64>>>(
            lossFunctionPtr_, ruleEvaluationFactoryPtr_, labelMatrixPtr_, std::move(statisticMatrixPtr),
            std::move(scoreMatrixPtr));
    }

}
