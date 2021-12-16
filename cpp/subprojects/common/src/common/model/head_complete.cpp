#include "common/model/head_complete.hpp"
#include "common/data/arrays.hpp"


CompleteHead::CompleteHead(uint32 numElements)
    : numElements_(numElements), scores_(new float64[numElements]) {

}

CompleteHead::CompleteHead(const CompletePrediction& prediction)
    : CompleteHead(prediction.getNumElements()) {
    copyArray(prediction.scores_cbegin(), scores_, numElements_);
}

CompleteHead::~CompleteHead() {
    delete[] scores_;
}

uint32 CompleteHead::getNumElements() const {
    return numElements_;
}

CompleteHead::score_iterator CompleteHead::scores_begin() {
    return scores_;
}

CompleteHead::score_iterator CompleteHead::scores_end() {
    return &scores_[numElements_];
}

CompleteHead::score_const_iterator CompleteHead::scores_cbegin() const {
    return scores_;
}

CompleteHead::score_const_iterator CompleteHead::scores_cend() const {
    return &scores_[numElements_];
}

void CompleteHead::visit(CompleteHeadVisitor completeHeadVisitor, PartialHeadVisitor partialHeadVisitor) const {
    completeHeadVisitor(*this);
}
