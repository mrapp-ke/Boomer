/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss.hpp"
#include "common/prediction/probability_calibration_joint.hpp"
#include "common/prediction/probability_calibration_marginal.hpp"
#include "common/sampling/partition_sampling.hpp"
#include "common/stopping/global_pruning.hpp"

namespace boosting {

    /**
     * Allows to configure a method that automatically decides for a method that partitions the available training
     * examples into a training set and a holdout set, depending on whether a holdout set is needed and depending on the
     * loss function.
     */
    class AutomaticPartitionSamplingConfig final : public IPartitionSamplingConfig {
        private:

            const std::unique_ptr<IGlobalPruningConfig>& globalPruningConfigPtr_;

            const std::unique_ptr<IMarginalProbabilityCalibratorConfig>& marginalProbabilityCalibratorConfigPtr_;

            const std::unique_ptr<IJointProbabilityCalibratorConfig>& jointProbabilityCalibratorConfigPtr_;

        public:

            /**
             * @param globalPruningConfigPtr                    A reference to an unique pointer that stores the
             *                                                  configuration of the method that is used for pruning
             *                                                  entire rules
             * @param marginalProbabilityCalibratorConfigPtr    A reference to an unique pointer that stores the
             *                                                  configuration of the calibrator that is used to fit a
             *                                                  model for the calibration of marginal probabilities
             * @param jointProbabilityCalibratorConfigPtr       A reference to an unique pointer that stores the
             *                                                  configuration of the calibrator that is used to fit a
             *                                                  model for the calibration of joint probabilities
             */
            AutomaticPartitionSamplingConfig(
              const std::unique_ptr<IGlobalPruningConfig>& globalPruningConfigPtr,
              const std::unique_ptr<IMarginalProbabilityCalibratorConfig>& marginalProbabilityCalibratorConfigPtr,
              const std::unique_ptr<IJointProbabilityCalibratorConfig>& jointProbabilityCalibratorConfigPtr);

            /**
             * @see `IPartitionSamplingConfig::createPartitionSamplingFactory`
             */
            std::unique_ptr<IPartitionSamplingFactory> createPartitionSamplingFactory() const override;
    };

}
