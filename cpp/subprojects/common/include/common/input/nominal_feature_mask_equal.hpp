/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/nominal_feature_mask.hpp"
#include "common/macros.hpp"
#include <memory>


/**
 * Defines an interface for all classes that allow to check whether individual features are nominal or not in cases
 * where all features are of the same type, i.e., where all features are either nominal or numerical/ordinal.
 */
class MLRLCOMMON_API IEqualNominalFeatureMask : public INominalFeatureMask {

    public:

        virtual ~IEqualNominalFeatureMask() override { };

};

/**
 * Creates and returns a new object of type `IEqualNominalFeatureMask`.
 *
 * @param nominal   True, if all features are nominal, false, if all features are numerical/ordinal
 * @return          An unique pointer to an object of type `IEqualNominalFeatureMask` that has been created
 */
MLRLCOMMON_API std::unique_ptr<IEqualNominalFeatureMask> createEqualNominalFeatureMask(bool nominal);
