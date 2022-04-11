#include "common/output/label_space_info_no.hpp"
#include "common/output/predictor_classification.hpp"
#include "common/output/predictor_regression.hpp"
#include "common/output/predictor_probability.hpp"
#include "common/model/rule_list.hpp"


/**
 * An implementation of the type `INoLabelSpaceInfo` that does not provide any information about the label space.
 */
class NoLabelSpaceInfo final : public INoLabelSpaceInfo {

    public:

        std::unique_ptr<IClassificationPredictor> createClassificationPredictor(
                const IClassificationPredictorFactory& factory, const RuleList& model) const override {
            return factory.create(model, nullptr);
        }

        std::unique_ptr<IRegressionPredictor> createRegressionPredictor(
                const IRegressionPredictorFactory& factory, const RuleList& model) const override {
            return factory.create(model, nullptr);
        }

        std::unique_ptr<IProbabilityPredictor> createProbabilityPredictor(
                const IProbabilityPredictorFactory& factory, const RuleList& model) const override {
            return factory.create(model, nullptr);
        }

};

std::unique_ptr<INoLabelSpaceInfo> createNoLabelSpaceInfo() {
    return std::make_unique<NoLabelSpaceInfo>();
}
