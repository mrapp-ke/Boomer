/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/pruning/pruning.hpp"


/**
 * Allows to configure a strategy for pruning classification rules that prunes rules by following the ideas of
 * "incremental reduced error pruning" (IREP).
 */
class IrepConfig final : public IPruningConfig {

    public:

        std::unique_ptr<IPruningFactory> createPruningFactory() const override;

};
