#include "boosting/losses/loss_example_wise_squared_hinge.hpp"

#include "common/iterator/binary_forward_iterator.hpp"
#include "common/math/math.hpp"

namespace boosting {

    template<typename LabelIterator>
    static inline void updateLabelWiseStatisticsInternally(VectorConstView<float64>::const_iterator scoreIterator,
                                                           LabelIterator labelIterator,
                                                           DenseLabelWiseStatisticView::iterator statisticIterator,
                                                           uint32 numLabels) {
        LabelIterator labelIterator2 = labelIterator;

        // For each label `i`, calculate `x_i = predictedScore_i^2 - 2 * predictedScore_i + 1` if trueLabel_i = 1 and
        // `predictedScore_i < 1` or `x_i = predictedScore^2` if `trueLabel_i = 0` and `predictedScore_i > 0`
        // or `x_i = 0` otherwise. The of those values is used as a denominator when calculating the gradients and
        // Hessians afterwards...
        float64 denominator = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator;
            float64 x;

            if (trueLabel) {
                if (predictedScore < 1) {
                    x = (predictedScore * predictedScore) - (2 * predictedScore) + 1;
                } else {
                    x = 0;
                }
            } else {
                if (predictedScore > 0) {
                    x = (predictedScore * predictedScore);
                } else {
                    x = 0;
                }
            }

            statisticIterator[i].first = x;  // Temporarily store `x` in the array of gradients
            denominator += x;
            labelIterator++;
        }

        // The denominator that is used for the calculation of gradients is `sqrt(x_1 + x_2 + ...)`...
        float64 denominatorGradient = std::sqrt(denominator);

        // The denominator that is used for the calculation of Hessians is `(x_1 + x_2 + ...)^1.5`...
        float64 denominatorHessian = std::pow(denominator, 1.5);

        // Calculate the gradients and Hessians...
        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator2;
            Tuple<float64>& tuple = statisticIterator[i];
            float64 gradient;
            float64 hessian;

            if (trueLabel) {
                if (predictedScore < 1) {
                    gradient = divideOrZero<float64>(predictedScore - 1, denominatorGradient);
                    hessian = divideOrZero<float64>(denominator - tuple.first, denominatorHessian);
                } else {
                    gradient = 0;
                    hessian = 1;
                }
            } else {
                if (predictedScore > 0) {
                    gradient = divideOrZero<float64>(predictedScore, denominatorGradient);
                    hessian = divideOrZero<float64>(denominator - tuple.first, denominatorHessian);
                } else {
                    gradient = 0;
                    hessian = 1;
                }
            }

            tuple.first = gradient;
            tuple.second = hessian;
            labelIterator2++;
        }
    }

    template<typename LabelIterator>
    static inline void updateExampleWiseStatisticsInternally(
      VectorConstView<float64>::const_iterator scoreIterator, LabelIterator labelIterator,
      DenseExampleWiseStatisticView::gradient_iterator gradientIterator,
      DenseExampleWiseStatisticView::hessian_iterator hessianIterator, uint32 numLabels) {
        LabelIterator labelIterator2 = labelIterator;
        LabelIterator labelIterator3 = labelIterator;

        // For each label `i`, calculate `x_i = predictedScore_i^2 - 2 * predictedScore_i + 1` if trueLabel_i = 1 and
        // `predictedScore_i < 1` or `x_i = predictedScore^2` if `trueLabel_i = 0` and `predictedScore_i > 0`
        // or `x_i = 0` otherwise. The of those values is used as a denominator when calculating the gradients and
        // Hessians afterwards...
        float64 denominator = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator;
            float64 x;

            if (trueLabel) {
                if (predictedScore < 1) {
                    x = (predictedScore * predictedScore) - (2 * predictedScore) + 1;
                } else {
                    x = 0;
                }
            } else {
                if (predictedScore > 0) {
                    x = (predictedScore * predictedScore);
                } else {
                    x = 0;
                }
            }

            gradientIterator[i] = x;  // Temporarily store `x` in the array of gradients
            denominator += x;
            labelIterator++;
        }

        // The denominator that is used for the calculation of gradients is `sqrt(x_1 + x_2 + ...)`...
        float64 denominatorGradient = std::sqrt(denominator);

        // The denominator that is used for the calculation of Hessians is `(x_1 + x_2 + ...)^1.5`...
        float64 denominatorHessian = std::pow(denominator, 1.5);

        // Calculate the gradients and Hessians...
        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator2;
            float64 gradient;
            float64 hessian;

            if (trueLabel) {
                if (predictedScore < 1) {
                    gradient = divideOrZero<float64>(predictedScore - 1, denominatorGradient);
                    hessian = divideOrZero<float64>(denominator - gradientIterator[i], denominatorHessian);
                } else {
                    gradient = 0;
                    hessian = 1;
                }
            } else {
                if (predictedScore > 0) {
                    gradient = divideOrZero<float64>(predictedScore, denominatorGradient);
                    hessian = divideOrZero<float64>(denominator - gradientIterator[i], denominatorHessian);
                } else {
                    gradient = 0;
                    hessian = 1;
                }
            }

            LabelIterator labelIterator4 = labelIterator3;

            for (uint32 j = 0; j < i; j++) {
                float64 hessianTriangle;

                if (gradient != 0) {
                    bool trueLabel2 = *labelIterator4;
                    float64 predictedScore2 = scoreIterator[j];
                    float64 numerator;

                    if (trueLabel2) {
                        if (predictedScore2 < 1) {
                            numerator = predictedScore2 - 1;
                        } else {
                            numerator = 0;
                        }
                    } else {
                        if (predictedScore2 > 0) {
                            numerator = predictedScore2;
                        } else {
                            numerator = 0;
                        }
                    }

                    if (trueLabel) {
                        numerator *= -(predictedScore - 1);
                    } else {
                        numerator *= -predictedScore;
                    }

                    hessianTriangle = divideOrZero<float64>(numerator, denominatorHessian);
                } else {
                    hessianTriangle = 0;
                }

                *hessianIterator = hessianTriangle;
                hessianIterator++;
                labelIterator4++;
            }

            gradientIterator[i] = gradient;
            *hessianIterator = hessian;
            hessianIterator++;
            labelIterator2++;
        }
    }

    template<typename LabelIterator>
    static inline float64 evaluateInternally(VectorConstView<float64>::const_iterator scoreIterator,
                                             LabelIterator labelIterator, uint32 numLabels) {
        // The example-wise squared hinge loss calculates as `sqrt((L_1 + ...)` with
        // `L_i = max(1 - predictedScore_i, 0)^2` if `trueLabel_i = 1` or `L_i = max(predictedScore_i, 0)^2` if
        // `trueLabel_i = 0`.
        float64 sumOfSquares = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            float64 predictedScore = scoreIterator[i];
            bool trueLabel = *labelIterator;

            if (trueLabel) {
                if (predictedScore < 1) {
                    sumOfSquares += ((1 - predictedScore) * (1 - predictedScore));
                }
            } else {
                if (predictedScore > 0) {
                    sumOfSquares += (predictedScore * predictedScore);
                }
            }

            labelIterator++;
        }

        return std::sqrt(sumOfSquares);
    }

    /**
     * An implementation of the type `IExampleWiseLoss` that implements a multi-label variant of the squared hinge loss
     * that is applied example-wise.
     */
    class ExampleWiseSquaredHingeLoss final : public IExampleWiseLoss {
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
                uint32 numLabels = scoresEnd - scoresBegin;
                auto labelIterator = make_binary_forward_iterator(labelVector.cbegin(), labelVector.cend());
                return evaluateInternally(scoresBegin, labelIterator, numLabels);
            }
    };

    /**
     * Allows to create instances of the type `IExampleWiseLoss` that implement a multi-label variant of the squared
     * hinge loss that is applied example-wise.
     */
    class ExampleWiseSquaredHingeLossFactory final : public IExampleWiseLossFactory {
        public:

            std::unique_ptr<IExampleWiseLoss> createExampleWiseLoss() const override {
                return std::make_unique<ExampleWiseSquaredHingeLoss>();
            }
    };

    ExampleWiseSquaredHingeLossConfig::ExampleWiseSquaredHingeLossConfig(
      const std::unique_ptr<IHeadConfig>& headConfigPtr)
        : headConfigPtr_(headConfigPtr) {}

    std::unique_ptr<IStatisticsProviderFactory> ExampleWiseSquaredHingeLossConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
      const Lapack& lapack, bool preferSparseStatistics) const {
        return headConfigPtr_->createStatisticsProviderFactory(featureMatrix, labelMatrix, *this, blas, lapack);
    }

    std::unique_ptr<IMarginalProbabilityFunctionFactory>
      ExampleWiseSquaredHingeLossConfig::createMarginalProbabilityFunctionFactory() const {
        return nullptr;
    }

    std::unique_ptr<IJointProbabilityFunctionFactory>
      ExampleWiseSquaredHingeLossConfig::createJointProbabilityFunctionFactory() const {
        return nullptr;
    }

    float64 ExampleWiseSquaredHingeLossConfig::getDefaultPrediction() const {
        return 0.5;
    }

    std::unique_ptr<IExampleWiseLossFactory> ExampleWiseSquaredHingeLossConfig::createExampleWiseLossFactory() const {
        return std::make_unique<ExampleWiseSquaredHingeLossFactory>();
    }

}