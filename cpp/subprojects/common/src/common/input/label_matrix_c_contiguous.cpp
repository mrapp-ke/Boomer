#include "common/input/label_matrix_c_contiguous.hpp"

#include "common/math/math.hpp"
#include "common/prediction/probability_calibration_joint.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/sampling/partition_sampling.hpp"
#include "common/statistics/statistics_provider.hpp"

CContiguousLabelMatrix::View::View(const CContiguousLabelMatrix& labelMatrix, uint32 row)
    : VectorConstView<const uint8>(labelMatrix.getNumCols(), labelMatrix.values_cbegin(row)) {}

CContiguousLabelMatrix::CContiguousLabelMatrix(uint32 numRows, uint32 numCols, const uint8* array)
    : CContiguousConstView<const uint8>(numRows, numCols, array) {}

bool CContiguousLabelMatrix::isSparse() const {
    return false;
}

float32 CContiguousLabelMatrix::calculateLabelCardinality() const {
    uint32 numRows = this->getNumRows();
    uint32 numCols = this->getNumCols();
    float32 labelCardinality = 0;

    for (uint32 i = 0; i < numRows; i++) {
        value_const_iterator labelIterator = this->values_cbegin(i);
        uint32 numRelevantLabels = 0;

        for (uint32 j = 0; j < numCols; j++) {
            if (labelIterator[j]) {
                numRelevantLabels++;
            }
        }

        labelCardinality = iterativeArithmeticMean(i + 1, (float32) numRelevantLabels, labelCardinality);
    }

    return labelCardinality;
}

CContiguousLabelMatrix::view_type CContiguousLabelMatrix::createView(uint32 row) const {
    return CContiguousLabelMatrix::view_type(*this, row);
}

std::unique_ptr<LabelVector> CContiguousLabelMatrix::createLabelVector(uint32 row) const {
    uint32 numCols = this->getNumCols();
    std::unique_ptr<LabelVector> labelVectorPtr = std::make_unique<LabelVector>(numCols);
    LabelVector::iterator iterator = labelVectorPtr->begin();
    value_const_iterator labelIterator = this->values_cbegin(row);
    uint32 n = 0;

    for (uint32 i = 0; i < numCols; i++) {
        if (labelIterator[i]) {
            iterator[n] = i;
            n++;
        }
    }

    labelVectorPtr->setNumElements(n, true);
    return labelVectorPtr;
}

std::unique_ptr<IStatisticsProvider> CContiguousLabelMatrix::createStatisticsProvider(
  const IStatisticsProviderFactory& factory) const {
    return factory.create(*this);
}

std::unique_ptr<IPartitionSampling> CContiguousLabelMatrix::createPartitionSampling(
  const IPartitionSamplingFactory& factory) const {
    return factory.create(*this);
}

std::unique_ptr<IInstanceSampling> CContiguousLabelMatrix::createInstanceSampling(
  const IInstanceSamplingFactory& factory, const SinglePartition& partition, IStatistics& statistics) const {
    return factory.create(*this, partition, statistics);
}

std::unique_ptr<IInstanceSampling> CContiguousLabelMatrix::createInstanceSampling(
  const IInstanceSamplingFactory& factory, BiPartition& partition, IStatistics& statistics) const {
    return factory.create(*this, partition, statistics);
}

std::unique_ptr<IMarginalProbabilityCalibrationModel> CContiguousLabelMatrix::fitMarginalProbabilityCalibrationModel(
  const IMarginalProbabilityCalibrator& probabilityCalibrator, const SinglePartition& partition,
  const IStatistics& statistics) const {
    return probabilityCalibrator.fitProbabilityCalibrationModel(partition, *this, statistics);
}

std::unique_ptr<IMarginalProbabilityCalibrationModel> CContiguousLabelMatrix::fitMarginalProbabilityCalibrationModel(
  const IMarginalProbabilityCalibrator& probabilityCalibrator, BiPartition& partition,
  const IStatistics& statistics) const {
    return probabilityCalibrator.fitProbabilityCalibrationModel(partition, *this, statistics);
}

std::unique_ptr<IJointProbabilityCalibrationModel> CContiguousLabelMatrix::fitJointProbabilityCalibrationModel(
  const IJointProbabilityCalibrator& probabilityCalibrator, const SinglePartition& partition,
  const IStatistics& statistics) const {
    return probabilityCalibrator.fitProbabilityCalibrationModel(partition, *this, statistics);
}

std::unique_ptr<IJointProbabilityCalibrationModel> CContiguousLabelMatrix::fitJointProbabilityCalibrationModel(
  const IJointProbabilityCalibrator& probabilityCalibrator, BiPartition& partition,
  const IStatistics& statistics) const {
    return probabilityCalibrator.fitProbabilityCalibrationModel(partition, *this, statistics);
}

std::unique_ptr<ICContiguousLabelMatrix> createCContiguousLabelMatrix(uint32 numRows, uint32 numCols,
                                                                      const uint8* array) {
    return std::make_unique<CContiguousLabelMatrix>(numRows, numCols, array);
}
