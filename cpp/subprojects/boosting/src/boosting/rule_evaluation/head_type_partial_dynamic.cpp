#include "boosting/rule_evaluation/head_type_partial_dynamic.hpp"

#include "boosting/statistics/statistics_provider_example_wise_dense.hpp"
#include "boosting/statistics/statistics_provider_label_wise_dense.hpp"
#include "boosting/statistics/statistics_provider_label_wise_sparse.hpp"
#include "common/util/validation.hpp"

namespace boosting {

    DynamicPartialHeadConfig::DynamicPartialHeadConfig(
      const std::unique_ptr<ILabelBinningConfig>& labelBinningConfigPtr,
      const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr)
        : threshold_(0.02f), exponent_(2.0f), labelBinningConfigPtr_(labelBinningConfigPtr),
          multiThreadingConfigPtr_(multiThreadingConfigPtr) {}

    float32 DynamicPartialHeadConfig::getThreshold() const {
        return threshold_;
    }

    IDynamicPartialHeadConfig& DynamicPartialHeadConfig::setThreshold(float32 threshold) {
        assertGreater<float32>("threshold", threshold, 0);
        assertLess<float32>("threshold", threshold, 1);
        threshold_ = threshold;
        return *this;
    }

    float32 DynamicPartialHeadConfig::getExponent() const {
        return exponent_;
    }

    IDynamicPartialHeadConfig& DynamicPartialHeadConfig::setExponent(float32 exponent) {
        assertGreaterOrEqual<float32>("exponent", exponent, 1);
        exponent_ = exponent;
        return *this;
    }

    std::unique_ptr<IStatisticsProviderFactory> DynamicPartialHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const ILabelWiseLossConfig& lossConfig) const {
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
        std::unique_ptr<ILabelWiseLossFactory> lossFactoryPtr = lossConfig.createLabelWiseLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createEvaluationMeasureFactory();
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createLabelWiseCompleteRuleEvaluationFactory();
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createLabelWiseDynamicPartialRuleEvaluationFactory(threshold_, exponent_);
        std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createLabelWiseDynamicPartialRuleEvaluationFactory(threshold_, exponent_);
        return std::make_unique<DenseLabelWiseStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IStatisticsProviderFactory> DynamicPartialHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const ISparseLabelWiseLossConfig& lossConfig) const {
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
        std::unique_ptr<ISparseLabelWiseLossFactory> lossFactoryPtr = lossConfig.createSparseLabelWiseLossFactory();
        std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createSparseEvaluationMeasureFactory();
        std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createLabelWiseDynamicPartialRuleEvaluationFactory(threshold_, exponent_);
        std::unique_ptr<ISparseLabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createLabelWiseDynamicPartialRuleEvaluationFactory(threshold_, exponent_);
        return std::make_unique<SparseLabelWiseStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(regularRuleEvaluationFactoryPtr),
          std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    std::unique_ptr<IStatisticsProviderFactory> DynamicPartialHeadConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
      const IExampleWiseLossConfig& lossConfig, const Blas& blas, const Lapack& lapack) const {
        uint32 numThreads = multiThreadingConfigPtr_->getNumThreads(featureMatrix, labelMatrix.getNumCols());
        std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr = lossConfig.createExampleWiseLossFactory();
        std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr =
          lossConfig.createExampleWiseLossFactory();
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createExampleWiseCompleteRuleEvaluationFactory(blas, lapack);
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createExampleWiseDynamicPartialRuleEvaluationFactory(threshold_, exponent_, blas,
                                                                                       lapack);
        std::unique_ptr<IExampleWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr =
          labelBinningConfigPtr_->createExampleWiseDynamicPartialRuleEvaluationFactory(threshold_, exponent_, blas,
                                                                                       lapack);
        return std::make_unique<DenseExampleWiseStatisticsProviderFactory>(
          std::move(lossFactoryPtr), std::move(evaluationMeasureFactoryPtr), std::move(defaultRuleEvaluationFactoryPtr),
          std::move(regularRuleEvaluationFactoryPtr), std::move(pruningRuleEvaluationFactoryPtr), numThreads);
    }

    bool DynamicPartialHeadConfig::isPartial() const {
        return true;
    }

    bool DynamicPartialHeadConfig::isSingleLabel() const {
        return false;
    }

}
