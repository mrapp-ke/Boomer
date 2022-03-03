#include "boosting/losses/loss_label_wise_squared_error.hpp"
#include "loss_label_wise_common.hpp"


namespace boosting {

    static inline void updateGradientAndHessian(bool trueLabel, float64 predictedScore, float64* gradient,
                                                float64* hessian) {
        float64 expectedScore = trueLabel ? 1 : -1;
        *gradient = (predictedScore - expectedScore);
        *hessian = 1;
    }

    static inline float64 evaluatePrediction(bool trueLabel, float64 predictedScore) {
        float64 expectedScore = trueLabel ? 1 : -1;
        float64 difference = (expectedScore - predictedScore);
        return difference * difference;
    }

    /**
     * Allows to create instances of the type `ILabelWiseLoss` that implement a multi-label variant of the squared error
     * loss that is applied label-wise.
     */
    class LabelWiseSquaredErrorLossFactory final : public ILabelWiseLossFactory {

        public:

            std::unique_ptr<ILabelWiseLoss> createLabelWiseLoss() const override {
                return std::make_unique<LabelWiseLoss>(&updateGradientAndHessian, &evaluatePrediction);
            }

    };

    LabelWiseSquaredErrorLossConfig::LabelWiseSquaredErrorLossConfig(const std::unique_ptr<IHeadConfig>& headConfigPtr)
        : headConfigPtr_(headConfigPtr) {

    }

    std::unique_ptr<IStatisticsProviderFactory> LabelWiseSquaredErrorLossConfig::createStatisticsProviderFactory(
            const IFeatureMatrix& featureMatrix, const ILabelMatrix& labelMatrix, const Blas& blas,
            const Lapack& lapack) const {
        return headConfigPtr_->createStatisticsProviderFactory(featureMatrix, labelMatrix, *this);
    }

    std::unique_ptr<IProbabilityFunctionFactory> LabelWiseSquaredErrorLossConfig::createProbabilityFunctionFactory() const {
        return nullptr;
    }

    float64 LabelWiseSquaredErrorLossConfig::getDefaultPrediction() const {
        return 0;
    }

    std::unique_ptr<ILabelWiseLossFactory> LabelWiseSquaredErrorLossConfig::createLabelWiseLossFactory() const {
        return std::make_unique<LabelWiseSquaredErrorLossFactory>();
    }

}
