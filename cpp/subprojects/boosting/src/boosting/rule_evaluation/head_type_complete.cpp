#include "boosting/rule_evaluation/head_type_complete.hpp"
#include "boosting/rule_evaluation/rule_evaluation_label_wise_complete.hpp"
#include "boosting/statistics/statistics_provider_example_wise_dense.hpp"
#include "boosting/statistics/statistics_provider_label_wise_dense.hpp"


namespace boosting {

    CompleteHeadConfig::CompleteHeadConfig(const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr,
                                           const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr,
                                           const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
                                           const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr)
        : labelBinningConfigPtr_(labelBinningConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr),
          l1RegularizationConfigPtr_(l1RegularizationConfigPtr), l2RegularizationConfigPtr_(l2RegularizationConfigPtr) {

    }

    std::unique_ptr<IStatisticsProviderFactory> CompleteHeadConfig::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix,
            const ILabelWiseLossConfig& lossConfig) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
        std::unique_ptr<ILabelWiseLossFactory> lossFactoryPtr = lossConfig.createLabelWiseLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
            lossConfig.createEvaluationMeasureFactory();
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
            labelBinningConfigPtr_->createLabelWiseRuleEvaluationFactory();
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
            std::make_unique<LabelWiseCompleteRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
            std::make_unique<LabelWiseCompleteRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        return std::make_unique<DenseLabelWiseStatisticsProviderFactory>(
            std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
            std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
            std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IStatisticsProviderFactory> CompleteHeadConfig::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix,
            const IExampleWiseLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
        std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr = lossConfig.createExampleWiseLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
            lossConfig.createExampleWiseLossFactory();
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
            labelBinningConfigPtr_->createExampleWiseRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
            labelBinningConfigPtr_->createExampleWiseRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
            labelBinningConfigPtr_->createExampleWiseRuleEvaluationFactory(blas, lapack);
        return std::make_unique<DenseExampleWiseStatisticsProviderFactory>(
            std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr),
            std::move(defaultRuleEvaluationFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
            std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

}
