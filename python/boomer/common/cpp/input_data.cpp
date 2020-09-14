#include "input_data.h"


AbstractLabelMatrix::AbstractLabelMatrix(uint32 numExamples, uint32 numLabels) {
    numExamples_ = numExamples;
    numLabels_ = numLabels;
}

AbstractLabelMatrix::~AbstractLabelMatrix() {

}

AbstractRandomAccessLabelMatrix::AbstractRandomAccessLabelMatrix(uint32 numExamples, uint32 numLabels)
    : AbstractLabelMatrix(numExamples, numLabels) {

}

uint8 AbstractRandomAccessLabelMatrix::getLabel(uint32 exampleIndex, uint32 labelIndex) {
    return 0;
}

DenseLabelMatrixImpl::DenseLabelMatrixImpl(uint32 numExamples, uint32 numLabels, const uint8* y)
    : AbstractRandomAccessLabelMatrix(numExamples, numLabels) {
    y_ = y;
}

DenseLabelMatrixImpl::~DenseLabelMatrixImpl() {

}

uint8 DenseLabelMatrixImpl::getLabel(uint32 exampleIndex, uint32 labelIndex) {
    uint32 i = (exampleIndex * numLabels_) + labelIndex;
    return y_[i];
}

DokLabelMatrixImpl::DokLabelMatrixImpl(uint32 numExamples, uint32 numLabels,
                                       std::shared_ptr<BinaryDokMatrix> dokMatrixPtr)
    : AbstractRandomAccessLabelMatrix(numExamples, numLabels) {
    dokMatrixPtr_ = dokMatrixPtr;
}

DokLabelMatrixImpl::~DokLabelMatrixImpl() {

}

uint8 DokLabelMatrixImpl::getLabel(uint32 exampleIndex, uint32 labelIndex) {
    return dokMatrixPtr_.get()->getValue(exampleIndex, labelIndex);
}
