#include "common/model/head_partial.hpp"


PartialHead::PartialHead(uint32 numElements)
    : numElements_(numElements), scores_(new float64[numElements]), labelIndices_(new uint32[numElements]) {

}

PartialHead::PartialHead(const PartialPrediction& prediction)
    : PartialHead(prediction.getNumElements()) {
    std::copy(prediction.scores_cbegin(), prediction.scores_cend(), scores_);
    std::copy(prediction.indices_cbegin(), prediction.indices_cend(), labelIndices_);
}

PartialHead::~PartialHead() {
    delete[] scores_;
    delete[] labelIndices_;
}

uint32 PartialHead::getNumElements() const {
    return numElements_;
}

PartialHead::score_iterator PartialHead::scores_begin() {
    return scores_;
}

PartialHead::score_iterator PartialHead::scores_end() {
    return &scores_[numElements_];
}

PartialHead::score_const_iterator PartialHead::scores_cbegin() const {
    return scores_;
}

PartialHead::score_const_iterator PartialHead::scores_cend() const {
    return &scores_[numElements_];
}

PartialHead::index_iterator PartialHead::indices_begin() {
    return labelIndices_;
}

PartialHead::index_iterator PartialHead::indices_end() {
    return &labelIndices_[numElements_];
}

PartialHead::index_const_iterator PartialHead::indices_cbegin() const {
    return labelIndices_;
}

PartialHead::index_const_iterator PartialHead::indices_cend() const {
    return &labelIndices_[numElements_];
}

void PartialHead::visit(FullHeadVisitor fullHeadVisitor, PartialHeadVisitor partialHeadVisitor) const {
    partialHeadVisitor(*this);
}
