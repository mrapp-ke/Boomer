#include "common/rule_model_assemblage/default_rule.hpp"

DefaultRuleConfig::DefaultRuleConfig(bool useDefaultRule) : useDefaultRule_(useDefaultRule) {}

bool DefaultRuleConfig::isDefaultRuleUsed(const IRowWiseLabelMatrix& labelMatrix) const {
    return useDefaultRule_;
}
