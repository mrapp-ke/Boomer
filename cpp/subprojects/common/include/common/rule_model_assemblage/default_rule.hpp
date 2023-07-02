/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/label_matrix_row_wise.hpp"

/**
 * Defines an interface for all classes that allow to configure the default rule that is included in a rule-based model.
 */
class IDefaultRuleConfig {
    public:

        virtual ~IDefaultRuleConfig() {};

        /**
         * Returns whether a default rule is included or not.
         *
         * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access to
         *                      the labels of the training examples
         * @return              True, if a default rule is included, false otherwise
         */
        virtual bool isDefaultRuleUsed(const IRowWiseLabelMatrix& labelMatrix) const = 0;
};

/**
 * Allows to configure whether a default rule should be included in a rule-based model or not.
 */
class DefaultRuleConfig final : public IDefaultRuleConfig {
    private:

        const bool useDefaultRule_;

    public:

        /**
         * @param useDefaultRule True, if a default rule should be included, false otherwise
         */
        DefaultRuleConfig(bool useDefaultRule);

        bool isDefaultRuleUsed(const IRowWiseLabelMatrix& labelMatrix) const override;
};
