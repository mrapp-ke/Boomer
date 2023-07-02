#include "boosting/losses/loss_label_wise_squared_hinge.hpp"

#include "loss_label_wise_sparse_common.hpp"

namespace boosting {

    static inline void updateGradientAndHessian(bool trueLabel, float64 predictedScore, float64* gradient,
                                                float64* hessian) {
        if (trueLabel) {
            if (predictedScore < 1) {
                *gradient = (predictedScore - 1);
            } else {
                *gradient = 0;
            }
        } else {
            if (predictedScore > 0) {
                *gradient = predictedScore;
            } else {
                *gradient = 0;
            }
        }

        *hessian = 1;
    }

    static inline float64 evaluatePrediction(bool trueLabel, float64 predictedScore) {
        if (trueLabel) {
            if (predictedScore < 1) {
                return (1 - predictedScore) * (1 - predictedScore);
            } else {
                return 0;
            }
        } else {
            if (predictedScore > 0) {
                return predictedScore * predictedScore;
            } else {
                return 0;
            }
        }
    }

    /**
     * Allows to create instances of the type `ILabelWiseLoss` that implement a multi-label variant of the squared hinge
     * loss that is applied label-wise.
     */
    class LabelWiseSquaredHingeLossFactory final : public ISparseLabelWiseLossFactory {
        public:

            std::unique_ptr<ISparseLabelWiseLoss> createSparseLabelWiseLoss() const override {
                return std::make_unique<SparseLabelWiseLoss>(&updateGradientAndHessian, &evaluatePrediction);
            }
    };

    LabelWiseSquaredHingeLossConfig::LabelWiseSquaredHingeLossConfig(const std::unique_ptr<IHeadConfig>& headConfigPtr)
        : headConfigPtr_(headConfigPtr) {}

    std::unique_ptr<IStatisticsProviderFactory> LabelWiseSquaredHingeLossConfig::createStatisticsProviderFactory(
      const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
      const Lapack& lapack, bool preferSparseStatistics) const {
        if (preferSparseStatistics) {
            return headConfigPtr_->createStatisticsProviderFactory(featureMatrix, labelMatrix, *this);
        } else {
            return headConfigPtr_->createStatisticsProviderFactory(featureMatrix, labelMatrix,
                                                                   static_cast<const ILabelWiseLossConfig&>(*this));
        }
    }

    std::unique_ptr<IMarginalProbabilityFunctionFactory>
      LabelWiseSquaredHingeLossConfig::createMarginalProbabilityFunctionFactory() const {
        return nullptr;
    }

    std::unique_ptr<IJointProbabilityFunctionFactory>
      LabelWiseSquaredHingeLossConfig::createJointProbabilityFunctionFactory() const {
        return nullptr;
    }

    float64 LabelWiseSquaredHingeLossConfig::getDefaultPrediction() const {
        return 0.5;
    }

    std::unique_ptr<ISparseLabelWiseLossFactory> LabelWiseSquaredHingeLossConfig::createSparseLabelWiseLossFactory()
      const {
        return std::make_unique<LabelWiseSquaredHingeLossFactory>();
    }

}
