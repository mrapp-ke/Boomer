#include "common/model/head_full.hpp"


FullHead::FullHead(uint32 numElements)
    : numElements_(numElements), scores_(new float64[numElements]) {

}

FullHead::FullHead(const FullPrediction& prediction)
    : FullHead(prediction.getNumElements()) {
    std::copy(prediction.scores_cbegin(), prediction.scores_cend(), scores_);
}

FullHead::~FullHead() {
    delete[] scores_;
}

uint32 FullHead::getNumElements() const {
    return numElements_;
}

FullHead::score_iterator FullHead::scores_begin() {
    return scores_;
}

FullHead::score_iterator FullHead::scores_end() {
    return &scores_[numElements_];
}

FullHead::score_const_iterator FullHead::scores_cbegin() const {
    return scores_;
}

FullHead::score_const_iterator FullHead::scores_cend() const {
    return &scores_[numElements_];
}

void FullHead::visit(FullHeadVisitor fullHeadVisitor, PartialHeadVisitor partialHeadVisitor) const {
    fullHeadVisitor(*this);
}
