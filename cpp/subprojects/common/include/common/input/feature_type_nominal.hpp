/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_type.hpp"

/**
 * Represents a nominal feature.
 */
class NominalFeatureType final : public IFeatureType {
    public:

        bool isNominal() const override;
};
