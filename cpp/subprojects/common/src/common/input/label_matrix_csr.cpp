#include "common/input/label_matrix_csr.hpp"

#include "common/data/arrays.hpp"
#include "common/math/math.hpp"
#include "common/prediction/probability_calibration_joint.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/sampling/partition_sampling.hpp"
#include "common/statistics/statistics_provider.hpp"

CsrLabelMatrix::View::View(const CsrLabelMatrix& labelMatrix, uint32 row)
    : VectorConstView<const uint32>(labelMatrix.indices_cend(row) - labelMatrix.indices_cbegin(row),
                                    labelMatrix.indices_cbegin(row)) {}

CsrLabelMatrix::CsrLabelMatrix(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices)
    : BinaryCsrConstView(numRows, numCols, rowIndices, colIndices) {}

bool CsrLabelMatrix::isSparse() const {
    return true;
}

float32 CsrLabelMatrix::calculateLabelCardinality() const {
    uint32 numRows = this->getNumRows();
    float32 labelCardinality = 0;

    for (uint32 i = 0; i < numRows; i++) {
        index_const_iterator indicesBegin = this->indices_cbegin(i);
        index_const_iterator indicesEnd = this->indices_cend(i);
        uint32 numRelevantLabels = indicesEnd - indicesBegin;
        labelCardinality = iterativeArithmeticMean(i + 1, (float32) numRelevantLabels, labelCardinality);
    }

    return labelCardinality;
}

CsrLabelMatrix::view_type CsrLabelMatrix::createView(uint32 row) const {
    return CsrLabelMatrix::view_type(*this, row);
}

std::unique_ptr<LabelVector> CsrLabelMatrix::createLabelVector(uint32 row) const {
    index_const_iterator indexIterator = this->indices_cbegin(row);
    index_const_iterator indicesEnd = this->indices_cend(row);
    uint32 numElements = indicesEnd - indexIterator;
    std::unique_ptr<LabelVector> labelVectorPtr = std::make_unique<LabelVector>(numElements);
    LabelVector::iterator iterator = labelVectorPtr->begin();
    copyArray(indexIterator, iterator, numElements);
    return labelVectorPtr;
}

std::unique_ptr<IStatisticsProvider> CsrLabelMatrix::createStatisticsProvider(
  const IStatisticsProviderFactory& factory) const {
    return factory.create(*this);
}

std::unique_ptr<IPartitionSampling> CsrLabelMatrix::createPartitionSampling(
  const IPartitionSamplingFactory& factory) const {
    return factory.create(*this);
}

std::unique_ptr<IInstanceSampling> CsrLabelMatrix::createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                          const SinglePartition& partition,
                                                                          IStatistics& statistics) const {
    return factory.create(*this, partition, statistics);
}

std::unique_ptr<IInstanceSampling> CsrLabelMatrix::createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                          BiPartition& partition,
                                                                          IStatistics& statistics) const {
    return factory.create(*this, partition, statistics);
}

std::unique_ptr<IMarginalProbabilityCalibrationModel> CsrLabelMatrix::fitMarginalProbabilityCalibrationModel(
  const IMarginalProbabilityCalibrator& probabilityCalibrator, const SinglePartition& partition,
  const IStatistics& statistics) const {
    return probabilityCalibrator.fitProbabilityCalibrationModel(partition, *this, statistics);
}

std::unique_ptr<IMarginalProbabilityCalibrationModel> CsrLabelMatrix::fitMarginalProbabilityCalibrationModel(
  const IMarginalProbabilityCalibrator& probabilityCalibrator, BiPartition& partition,
  const IStatistics& statistics) const {
    return probabilityCalibrator.fitProbabilityCalibrationModel(partition, *this, statistics);
}

std::unique_ptr<IJointProbabilityCalibrationModel> CsrLabelMatrix::fitJointProbabilityCalibrationModel(
  const IJointProbabilityCalibrator& probabilityCalibrator, const SinglePartition& partition,
  const IStatistics& statistics) const {
    return probabilityCalibrator.fitProbabilityCalibrationModel(partition, *this, statistics);
}

std::unique_ptr<IJointProbabilityCalibrationModel> CsrLabelMatrix::fitJointProbabilityCalibrationModel(
  const IJointProbabilityCalibrator& probabilityCalibrator, BiPartition& partition,
  const IStatistics& statistics) const {
    return probabilityCalibrator.fitProbabilityCalibrationModel(partition, *this, statistics);
}

std::unique_ptr<ICsrLabelMatrix> createCsrLabelMatrix(uint32 numRows, uint32 numCols, uint32* rowIndices,
                                                      uint32* colIndices) {
    return std::make_unique<CsrLabelMatrix>(numRows, numCols, rowIndices, colIndices);
}
