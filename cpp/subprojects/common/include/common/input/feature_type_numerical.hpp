/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_type.hpp"

/**
 * Represents a numerical/ordinal feature.
 */
class NumericalFeatureType final : public IFeatureType {
    public:

        bool isNominal() const override;
};
