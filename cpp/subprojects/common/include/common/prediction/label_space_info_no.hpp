/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/label_space_info.hpp"

/**
 * Defines an interface for all classes that do not provide any information about the label space.
 */
class MLRLCOMMON_API INoLabelSpaceInfo : public ILabelSpaceInfo {
    public:

        virtual ~INoLabelSpaceInfo() override {};
};

/**
 * Creates and returns a new object of the type `INoLabelSpaceInfo`.
 *
 * @return An unique pointer to an object of type `INoLabelSpaceInfo` that has been created
 */
MLRLCOMMON_API std::unique_ptr<INoLabelSpaceInfo> createNoLabelSpaceInfo();
