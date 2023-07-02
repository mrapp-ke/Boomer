/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_type.hpp"

/**
 * Represents an ordinal feature.
 */
class OrdinalFeatureType final : public IFeatureType {
    public:

        bool isNominal() const override;
};
