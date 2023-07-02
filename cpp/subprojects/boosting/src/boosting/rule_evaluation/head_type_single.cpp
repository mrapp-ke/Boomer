#include "boosting/rule_evaluation/head_type_single.hpp"

#include "boosting/rule_evaluation/rule_evaluation_label_wise_single.hpp"
#include "boosting/statistics/statistics_provider_example_wise_dense.hpp"
#include "boosting/statistics/statistics_provider_label_wise_dense.hpp"
#include "boosting/statistics/statistics_provider_label_wise_sparse.hpp"

namespace boosting {

    SingleLabelHeadConfig::SingleLabelHeadConfig(
      const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr,
      const std::unique_ptr<IRegularizationConfig>& l1RegularizationConfigPtr,
      const std::unique_ptr<IRegularizationConfig>& l2RegularizationConfigPtr)
        : labelBinningConfigPtr_(labelBinningConfigPtr), multiThreadingConfigPtr_(multiThreadingConfigPtr),
          l1RegularizationConfigPtr_(l1RegularizationConfigPtr), l2RegularizationConfigPtr_(l2RegularizationConfigPtr) {

    }

    std::unique_ptr<IStatisticsProviderFactory> SingleLabelHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const ILabelWiseLossConfig& lossConfig) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
        std::unique_ptr<ILabelWiseLossFactory> lossFactoryPtr = lossConfig.createLabelWiseLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createEvaluationMeasureFactory();
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createLabelWiseCompleteRuleEvaluationFactory();
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        return std::make_unique<DenseLabelWiseStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IStatisticsProviderFactory> SingleLabelHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const ISparseLabelWiseLossConfig& lossConfig) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
        std::unique_ptr<ISparseLabelWiseLossFactory> lossFactoryPtr = lossConfig.createSparseLabelWiseLossFactory();
        std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createSparseEvaluationMeasureFactory();
        std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        return std::make_unique<SparseLabelWiseStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
          std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IStatisticsProviderFactory> SingleLabelHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const IExampleWiseLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        float64 l1RegularizationWeight = l1RegularizationConfigPtr_->getWeight();
        float64 l2RegularizationWeight = l2RegularizationConfigPtr_->getWeight();
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
        std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr = lossConfig.createExampleWiseLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createExampleWiseLossFactory();
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createExampleWiseCompleteRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          std::make_unique<LabelWiseSingleLabelRuleEvaluationFactory>(l1RegularizationWeight, l2RegularizationWeight);
        return std::make_unique<DenseConvertibleExampleWiseStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    bool SingleLabelHeadConfig::isPartial() const {
        return true;
    }

    bool SingleLabelHeadConfig::isSingleLabel() const {
        return true;
    }

}
