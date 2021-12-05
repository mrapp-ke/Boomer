#include "common/input/label_matrix_csr.hpp"
#include "common/statistics/statistics_provider_factory.hpp"
#include "common/sampling/partition_sampling.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/data/arrays.hpp"
#include "common/math/math.hpp"


CsrLabelMatrix::View::View(const CsrLabelMatrix& labelMatrix, uint32 row)
    : VectorConstView<const uint32>(labelMatrix.view_.row_indices_cend(row) - labelMatrix.view_.row_indices_cbegin(row),
                                    labelMatrix.view_.row_indices_cbegin(row)) {

}

CsrLabelMatrix::CsrLabelMatrix(uint32 numRows, uint32 numCols, uint32* rowIndices, uint32* colIndices)
    : view_(BinaryCsrConstView(numRows, numCols, rowIndices, colIndices)) {

}

CsrLabelMatrix::index_const_iterator CsrLabelMatrix::row_indices_cbegin(uint32 row) const {
    return view_.row_indices_cbegin(row);
}

CsrLabelMatrix::index_const_iterator CsrLabelMatrix::row_indices_cend(uint32 row) const {
    return view_.row_indices_cend(row);
}

CsrLabelMatrix::value_const_iterator CsrLabelMatrix::row_values_cbegin(uint32 row) const {
    return view_.row_values_cbegin(row);
}

CsrLabelMatrix::value_const_iterator CsrLabelMatrix::row_values_cend(uint32 row) const {
    return view_.row_values_cend(row);
}

uint32 CsrLabelMatrix::getNumNonZeroElements() const {
    return view_.getNumNonZeroElements();
}

uint32 CsrLabelMatrix::getNumRows() const {
    return view_.getNumRows();
}

uint32 CsrLabelMatrix::getNumCols() const {
    return view_.getNumCols();
}

float64 CsrLabelMatrix::calculateLabelCardinality() const {
    uint32 numRows = this->getNumRows();
    float64 labelCardinality = 0;

    for (uint32 i = 0; i < numRows; i++) {
        index_const_iterator indicesBegin = this->row_indices_cbegin(i);
        index_const_iterator indicesEnd = this->row_indices_cend(i);
        uint32 numRelevantLabels = indicesEnd - indicesBegin;
        labelCardinality = iterativeArithmeticMean(i + 1, (float64) numRelevantLabels, labelCardinality);
    }

    return labelCardinality;
}

CsrLabelMatrix::view_type CsrLabelMatrix::createView(uint32 row) const {
    return CsrLabelMatrix::view_type(*this, row);
}

std::unique_ptr<LabelVector> CsrLabelMatrix::createLabelVector(uint32 row) const {
    index_const_iterator indexIterator = this->row_indices_cbegin(row);
    index_const_iterator indicesEnd = this->row_indices_cend(row);
    uint32 numElements = indicesEnd - indexIterator;
    std::unique_ptr<LabelVector> labelVectorPtr = std::make_unique<LabelVector>(numElements);
    LabelVector::index_iterator iterator = labelVectorPtr->indices_begin();
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
