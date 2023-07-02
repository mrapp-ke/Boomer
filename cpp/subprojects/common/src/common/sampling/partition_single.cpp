#include "common/sampling/partition_single.hpp"

#include "common/prediction/probability_calibration_joint.hpp"
#include "common/rule_refinement/prediction.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/stopping/stopping_criterion.hpp"
#include "common/thresholds/thresholds_subset.hpp"

SinglePartition::SinglePartition(uint32 numElements) : numElements_(numElements) {}

SinglePartition::const_iterator SinglePartition::cbegin() const {
    return IndexIterator();
}

SinglePartition::const_iterator SinglePartition::cend() const {
    return IndexIterator(numElements_);
}

uint32 SinglePartition::getNumElements() const {
    return numElements_;
}

std::unique_ptr<IStoppingCriterion> SinglePartition::createStoppingCriterion(const IStoppingCriterionFactory& factory) {
    return factory.create(*this);
}

std::unique_ptr<IInstanceSampling> SinglePartition::createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                           const IRowWiseLabelMatrix& labelMatrix,
                                                                           IStatistics& statistics) {
    return labelMatrix.createInstanceSampling(factory, *this, statistics);
}

Quality SinglePartition::evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset,
                                             const ICoverageState& coverageState, const AbstractPrediction& head) {
    return coverageState.evaluateOutOfSample(thresholdsSubset, *this, head);
}

void SinglePartition::recalculatePrediction(const IThresholdsSubset& thresholdsSubset,
                                            const ICoverageState& coverageState, AbstractPrediction& head) {
    coverageState.recalculatePrediction(thresholdsSubset, *this, head);
}

std::unique_ptr<IMarginalProbabilityCalibrationModel> SinglePartition::fitMarginalProbabilityCalibrationModel(
  const IMarginalProbabilityCalibrator& probabilityCalibrator, const IRowWiseLabelMatrix& labelMatrix,
  const IStatistics& statistics) {
    return labelMatrix.fitMarginalProbabilityCalibrationModel(probabilityCalibrator, *this, statistics);
}

std::unique_ptr<IJointProbabilityCalibrationModel> SinglePartition::fitJointProbabilityCalibrationModel(
  const IJointProbabilityCalibrator& probabilityCalibrator, const IRowWiseLabelMatrix& labelMatrix,
  const IStatistics& statistics) {
    return labelMatrix.fitJointProbabilityCalibrationModel(probabilityCalibrator, *this, statistics);
}
