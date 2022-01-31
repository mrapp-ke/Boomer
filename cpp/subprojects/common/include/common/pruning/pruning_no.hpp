/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/pruning/pruning.hpp"


/**
 * Allows to configure a method for pruning classification rules that does not actually perform any pruning.
 */
class NoPruningConfig final : public IPruningConfig {

    public:

        std::unique_ptr<IPruningFactory> createPruningFactory() const override;

};
