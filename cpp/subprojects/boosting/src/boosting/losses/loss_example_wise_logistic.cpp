#include "boosting/losses/loss_example_wise_logistic.hpp"

#include "boosting/prediction/probability_function_chain_rule.hpp"
#include "boosting/prediction/probability_function_logistic.hpp"
#include "common/iterator/binary_forward_iterator.hpp"
#include "common/math/math.hpp"

namespace boosting {

    static inline void updateGradientAndHessian(float64 invertedExpectedScore, float64 x, float64 max, float64 sumExp,
                                                float64& gradient, float64& hessian) {
        // Calculate the gradient that corresponds to the current label. The gradient calculates as
        // `-expectedScore_c * exp(x_c) / (1 + exp(x_1) + exp(x_2) + ...)`, which can be rewritten as
        // `-expectedScore_c * (exp(x_c - max) / sumExp)`
        float64 xExp = std::exp(x - max);
        float64 tmp = divideOrZero<float64>(xExp, sumExp);
        gradient = invertedExpectedScore * tmp;

        // Calculate the Hessian on the diagonal of the Hessian matrix that corresponds to the current label. Such
        // Hessian calculates as `exp(x_c) * (1 + exp(x_1) + exp(x_2) + ...) / (1 + exp(x_1) + exp(x_2) + ...)^2`,
        // or as `(exp(x_c - max) / sumExp) * (1 - exp(x_c - max) / sumExp)`
        hessian = tmp * (1 - tmp);
    }

    template<typename LabelIterator>
    static inline void updateLabelWiseStatisticsInternally(VectorConstView<float64>::const_iterator scoreIterator,
                                                           LabelIterator labelIterator,
                                                           DenseLabelWiseStatisticView::iterator statisticIterator,
                                                           uint32 numLabels) {
        // This implementation uses the so-called "exp-normalize-trick" to increase numerical stability (see, e.g.,
        // https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/). It is based on rewriting a fraction
        // of the form `exp(x_1) / (exp(x_1) + exp(x_2) + ...)` as
        // `exp(x_1 - max) / (exp(x_1 - max) + exp(x_2 - max) + ...)`, where `max = max(x_1, x_2, ...)`. To be able to
        // exploit this equivalence for the calculation of gradients and Hessians, they are calculated as products of
        // fractions of the above form.
        LabelIterator labelIterator2 = labelIterator;

        // For each label `c`, calculate `x = -expectedScore_c * predictedScore_c` and find the largest and second
        // largest values (that must be greater than 0, because `exp(1) = 0`) among all of them...
        float64 max = 0;  // The largest value

        for (uint32 c = 0; c < numLabels; c++) {
            float64 predictedScore = scoreIterator[c];
            bool trueLabel = *labelIterator;
            float64 x = trueLabel ? -predictedScore : predictedScore;
            statisticIterator[c].first = x;  // Temporarily store `x` in the array of statistics

            if (x > max) {
                max = x;
            }

            labelIterator++;
        }

        // Calculate `sumExp = exp(0 - max) + exp(x_1 - max) + exp(x_2 - max) + ...`
        float64 sumExp = std::exp(0.0 - max);

        for (uint32 c = 0; c < numLabels; c++) {
            float64 x = statisticIterator[c].first;
            sumExp += std::exp(x - max);
        }

        // Calculate the gradients and Hessians...
        for (uint32 c = 0; c < numLabels; c++) {
            float64 predictedScore = scoreIterator[c];
            bool trueLabel = *labelIterator2;
            float64 invertedExpectedScore = trueLabel ? -1 : 1;
            float64 x = predictedScore * invertedExpectedScore;
            Tuple<float64>& tuple = statisticIterator[c];
            updateGradientAndHessian(invertedExpectedScore, x, max, sumExp, tuple.first, tuple.second);
            labelIterator2++;
        }
    }

    template<typename LabelIterator>
    static inline void updateExampleWiseStatisticsInternally(
      VectorConstView<float64>::const_iterator scoreIterator, LabelIterator labelIterator,
      DenseExampleWiseStatisticView::gradient_iterator gradientIterator,
      DenseExampleWiseStatisticView::hessian_iterator hessianIterator, uint32 numLabels) {
        // This implementation uses the so-called "exp-normalize-trick" to increase numerical stability (see, e.g.,
        // https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/). It is based on rewriting a fraction
        // of the form `exp(x_1) / (exp(x_1) + exp(x_2) + ...)` as
        // `exp(x_1 - max) / (exp(x_1 - max) + exp(x_2 - max) + ...)`, where `max = max(x_1, x_2, ...)`. To be able to
        // exploit this equivalence for the calculation of gradients and Hessians, they are calculated as products of
        // fractions of the above form.
        LabelIterator labelIterator2 = labelIterator;
        LabelIterator labelIterator3 = labelIterator;

        // For each label `c`, calculate `x = -expectedScore_c * predictedScore_c` and find the largest and second
        // largest values (that must be greater than 0, because `exp(1) = 0`) among all of them...
        float64 max = 0;   // The largest value
        float64 max2 = 0;  // The second largest value

        for (uint32 c = 0; c < numLabels; c++) {
            float64 predictedScore = scoreIterator[c];
            bool trueLabel = *labelIterator;
            float64 x = trueLabel ? -predictedScore : predictedScore;
            gradientIterator[c] = x;  // Temporarily store `x` in the array of gradients

            if (x > max) {
                max2 = max;
                max = x;
            } else if (x > max2) {
                max2 = x;
            }

            labelIterator++;
        }

        // In the following, the largest value the exponential function may be applied to is `max + max2`, which happens
        // when Hessians that belong to the upper triangle of the Hessian matrix are calculated...
        max2 += max;

        // Calculate `sumExp = exp(0 - max) + exp(x_1 - max) + exp(x_2 - max) + ...`
        float64 sumExp = std::exp(0.0 - max);
        float64 zeroExp = std::exp(0.0 - max2);
        float64 sumExp2 = zeroExp;

        for (uint32 c = 0; c < numLabels; c++) {
            float64 x = gradientIterator[c];
            sumExp += std::exp(x - max);
            sumExp2 += std::exp(x - max2);
        }

        // Calculate `zeroExp / sumExp2` (it is needed multiple times for calculating Hessians that belong to the upper
        // triangle of the Hessian matrix)...
        zeroExp = divideOrZero<float64>(zeroExp, sumExp2);

        // Calculate the gradients and Hessians...
        for (uint32 c = 0; c < numLabels; c++) {
            float64 predictedScore = scoreIterator[c];
            bool trueLabel = *labelIterator2;
            float64 invertedExpectedScore = trueLabel ? -1 : 1;
            float64 x = predictedScore * invertedExpectedScore;

            // Calculate the Hessians that belong to the part of the Hessian matrix' upper triangle that corresponds to
            // the current label. Such Hessian calculates as
            // `-expectedScore_c * expectedScore_r * exp(x_c + x_r) / (1 + exp(x_1) + exp(x_2) + ...)^2`, or as
            // `-expectedScore_c * expectedScore_r * (exp(x_c + x_r - max) / sumExp) * (exp(0 - max) / sumExp)`
            LabelIterator labelIterator4 = labelIterator3;

            for (uint32 r = 0; r < c; r++) {
                float64 predictedScore2 = scoreIterator[r];
                bool trueLabel2 = *labelIterator4;
                float64 expectedScore2 = trueLabel2 ? 1 : -1;
                float64 x2 = predictedScore2 * -expectedScore2;
                *hessianIterator = invertedExpectedScore * expectedScore2
                                   * divideOrZero<float64>(std::exp(x + x2 - max2), sumExp2) * zeroExp;
                hessianIterator++;
                labelIterator4++;
            }

            updateGradientAndHessian(invertedExpectedScore, x, max, sumExp, gradientIterator[c], *hessianIterator);
            hessianIterator++;
            labelIterator2++;
        }
    }

    template<typename LabelIterator>
    static inline float64 evaluateInternally(VectorConstView<float64>::const_iterator scoreIterator,
                                             LabelIterator labelIterator, uint32 numLabels) {
        // The example-wise logistic loss calculates as
        // `log(1 + exp(-expectedScore_1 * predictedScore_1) + ... + exp(-expectedScore_2 * predictedScore_2) + ...)`.
        // In the following, we exploit the identity
        // `log(exp(x_1) + exp(x_2) + ...) = max + log(exp(x_1 - max) + exp(x_2 - max) + ...)`, where
        // `max = max(x_1, x_2, ...)`, to increase numerical stability (see, e.g., section "Log-sum-exp for computing
        // the log-distribution" in https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/).
        LabelIterator labelIterator2 = labelIterator;
        float64 max = 0;

        // For each label `i`, calculate `x = -expectedScore_i * predictedScore_i` and find the largest value (that must
        // be greater than 0, because `exp(1) = 0`) among all of them...
        for (uint32 i = 0; i < numLabels; i++) {
            bool trueLabel = *labelIterator;
            float64 predictedScore = scoreIterator[i];
            float64 x = trueLabel ? -predictedScore : predictedScore;

            if (x > max) {
                max = x;
            }

            labelIterator++;
        }

        // Calculate the example-wise loss as `max + log(exp(0 - max) + exp(x_1 - max) + ...)`...
        float64 sumExp = std::exp(0 - max);

        for (uint32 i = 0; i < numLabels; i++) {
            bool trueLabel = *labelIterator2;
            float64 predictedScore = scoreIterator[i];
            float64 x = trueLabel ? -predictedScore : predictedScore;
            sumExp += std::exp(x - max);
            labelIterator2++;
        }

        return max + std::log(sumExp);
    }

    /**
     * An implementation of the type `IExampleWiseLoss` that implements a multi-label variant of the logistic loss that
     * is applied example-wise.
     */
    class ExampleWiseLogisticLoss final : public IExampleWiseLoss {
        public:

            virtual void updateLabelWiseStatistics(uint32 exampleIndex,
                                                   const CContiguousConstView<const uint8>& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override {
                updateLabelWiseStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex),
                                                    labelMatrix.values_cbegin(exampleIndex),
                                                    statisticView.begin(exampleIndex), labelMatrix.getNumCols());
            }

            virtual void updateLabelWiseStatistics(uint32 exampleIndex,
                                                   const CContiguousConstView<const uint8>& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override {
                updateLabelWiseStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex),
                                                    labelMatrix.values_cbegin(exampleIndex),
                                                    statisticView.begin(exampleIndex), labelMatrix.getNumCols());
            }

            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   CompleteIndexVector::const_iterator labelIndicesBegin,
                                                   CompleteIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override {
                auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                  labelMatrix.indices_cend(exampleIndex));
                updateLabelWiseStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), labelIterator,
                                                    statisticView.begin(exampleIndex), labelMatrix.getNumCols());
            }

            virtual void updateLabelWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                                   const CContiguousConstView<float64>& scoreMatrix,
                                                   PartialIndexVector::const_iterator labelIndicesBegin,
                                                   PartialIndexVector::const_iterator labelIndicesEnd,
                                                   DenseLabelWiseStatisticView& statisticView) const override {
                auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                  labelMatrix.indices_cend(exampleIndex));
                updateLabelWiseStatisticsInternally(scoreMatrix.values_cbegin(exampleIndex), labelIterator,
                                                    statisticView.begin(exampleIndex), labelMatrix.getNumCols());
            }

            void updateExampleWiseStatistics(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                                             const CContiguousConstView<float64>& scoreMatrix,
                                             DenseExampleWiseStatisticView& statisticView) const override {
                updateExampleWiseStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), labelMatrix.values_cbegin(exampleIndex),
                  statisticView.gradients_begin(exampleIndex), statisticView.hessians_begin(exampleIndex),
                  labelMatrix.getNumCols());
            }

            void updateExampleWiseStatistics(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                                             const CContiguousConstView<float64>& scoreMatrix,
                                             DenseExampleWiseStatisticView& statisticView) const override {
                auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                  labelMatrix.indices_cend(exampleIndex));
                updateExampleWiseStatisticsInternally(
                  scoreMatrix.values_cbegin(exampleIndex), labelIterator, statisticView.gradients_begin(exampleIndex),
                  statisticView.hessians_begin(exampleIndex), labelMatrix.getNumCols());
            }

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const CContiguousConstView<const uint8>& labelMatrix,
                             const CContiguousConstView<float64>& scoreMatrix) const override {
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex),
                                          labelMatrix.values_cbegin(exampleIndex), labelMatrix.getNumCols());
            }

            /**
             * @see `IEvaluationMeasure::evaluate`
             */
            float64 evaluate(uint32 exampleIndex, const BinaryCsrConstView& labelMatrix,
                             const CContiguousConstView<float64>& scoreMatrix) const override {
                auto labelIterator = make_binary_forward_iterator(labelMatrix.indices_cbegin(exampleIndex),
                                                                  labelMatrix.indices_cend(exampleIndex));
                return evaluateInternally(scoreMatrix.values_cbegin(exampleIndex), labelIterator,
                                          labelMatrix.getNumCols());
            }

            /**
             * @see `IDistanceMeasure::measureDistance`
             */
            float64 measureDistance(uint32 labelVectorIndex, const LabelVector& labelVector,
                                    VectorView<float64>::const_iterator scoresBegin,
                                    VectorView<float64>::const_iterator scoresEnd) const override {
                // The example-wise logistic loss calculates as
                // `log(1 + exp(-expectedScore_1 * predictedScore_1) + ... + exp(-expectedScore_2 * predictedScore_2)
                // + ...)`. In the following, we exploit the identity `log(exp(x_1) + exp(x_2) + ...) =
                // max + log(exp(x_1 - max) + exp(x_2 - max) + ...)`, where `max = max(x_1, x_2, ...)`, to increase
                // numerical stability (see, e.g., section "Log-sum-exp for computing the log-distribution" in
                // https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/).
                uint32 numLabels = scoresEnd - scoresBegin;
                auto labelIterator = make_binary_forward_iterator(labelVector.cbegin(), labelVector.cend());
                float64 max = 0;

                // For each label `i`, calculate `x = -expectedScore_i * predictedScore_i` and find the largest value
                // (that must be greater than 0, because `exp(1) = 0`) among all of them...
                for (uint32 i = 0; i < numLabels; i++) {
                    float64 predictedScore = scoresBegin[i];
                    bool trueLabel = *labelIterator;
                    float64 x = trueLabel ? -predictedScore : predictedScore;

                    if (x > max) {
                        max = x;
                    }

                    labelIterator++;
                }

                // Calculate the example-wise loss as `max + log(exp(0 - max) + exp(x_1 - max) + ...)`...
                float64 sumExp = std::exp(0 - max);
                labelIterator = make_binary_forward_iterator(labelVector.cbegin(), labelVector.cend());

                for (uint32 i = 0; i < numLabels; i++) {
                    float64 predictedScore = scoresBegin[i];
                    bool trueLabel = *labelIterator;
                    float64 x = trueLabel ? -predictedScore : predictedScore;
                    sumExp += std::exp(x - max);
                    labelIterator++;
                }

                return max + std::log(sumExp);
            }
    };

    /**
     * Allows to create instances of the type `IExampleWiseLoss` that implement a multi-label variant of the logistic
     * loss that is applied example-wise.
     */
    class ExampleWiseLogisticLossFactory final : public IExampleWiseLossFactory {
        public:

            std::unique_ptr<IExampleWiseLoss> createExampleWiseLoss() const override {
                return std::make_unique<ExampleWiseLogisticLoss>();
            }
    };

    ExampleWiseLogisticLossConfig::ExampleWiseLogisticLossConfig(const std::unique_ptr<IHeadConfig>& headConfigPtr)
        : headConfigPtr_(headConfigPtr) {}

    std::unique_ptr<IStatisticsProviderFactory> ExampleWiseLogisticLossConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
      const Lapack& lapack, bool preferSparseStatistics) const {
        return headConfigPtr_->createStatisticsProviderFactory(featureMatrix, labelMatrix, *this, blas, lapack);
    }

    std::unique_ptr<IMarginalProbabilityFunctionFactory>
      ExampleWiseLogisticLossConfig::createMarginalProbabilityFunctionFactory() const {
        return std::make_unique<LogisticFunctionFactory>();
    }

    std::unique_ptr<IJointProbabilityFunctionFactory>
      ExampleWiseLogisticLossConfig::createJointProbabilityFunctionFactory() const {
        return std::make_unique<ChainRuleFactory>(this->createMarginalProbabilityFunctionFactory());
    }

    float64 ExampleWiseLogisticLossConfig::getDefaultPrediction() const {
        return 0;
    }

    std::unique_ptr<IExampleWiseLossFactory> ExampleWiseLogisticLossConfig::createExampleWiseLossFactory() const {
        return std::make_unique<ExampleWiseLogisticLossFactory>();
    }

}