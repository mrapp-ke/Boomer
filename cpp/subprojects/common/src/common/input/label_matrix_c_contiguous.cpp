#include "common/input/label_matrix_c_contiguous.hpp"
#include "common/statistics/statistics_provider_factory.hpp"
#include "common/sampling/partition_sampling.hpp"
#include "common/sampling/instance_sampling.hpp"


CContiguousLabelMatrix::View::View(const CContiguousLabelMatrix& labelMatrix, uint32 row)
    : VectorConstView<const uint8>(labelMatrix.getNumCols(), labelMatrix.view_.row_cbegin(row)) {

}

CContiguousLabelMatrix::CContiguousLabelMatrix(uint32 numRows, uint32 numCols, const uint8* array)
    : view_(CContiguousConstView<const uint8>(numRows, numCols, array)) {

}

CContiguousLabelMatrix::value_const_iterator CContiguousLabelMatrix::row_values_cbegin(uint32 row) const {
    return view_.row_cbegin(row);
}

CContiguousLabelMatrix::value_const_iterator CContiguousLabelMatrix::row_values_cend(uint32 row) const {
    return view_.row_cend(row);
}

uint32 CContiguousLabelMatrix::getNumRows() const {
    return view_.getNumRows();
}

uint32 CContiguousLabelMatrix::getNumCols() const {
    return view_.getNumCols();
}

CContiguousLabelMatrix::view_type CContiguousLabelMatrix::createView(uint32 row) const {
    return CContiguousLabelMatrix::view_type(*this, row);
}

std::unique_ptr<LabelVector> CContiguousLabelMatrix::createLabelVector(uint32 row) const {
    uint32 numCols = this->getNumCols();
    std::unique_ptr<LabelVector> labelVectorPtr = std::make_unique<LabelVector>(numCols);
    LabelVector::index_iterator iterator = labelVectorPtr->indices_begin();
    value_const_iterator labelIterator = this->row_values_cbegin(row);
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
