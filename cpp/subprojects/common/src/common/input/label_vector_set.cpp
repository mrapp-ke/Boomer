#include "common/input/label_vector_set.hpp"


LabelVectorSet::const_iterator LabelVectorSet::cbegin() const {
    return labelVectors_.cbegin();
}

LabelVectorSet::const_iterator LabelVectorSet::cend() const {
    return labelVectors_.cend();
}

void LabelVectorSet::addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr) {
    ++labelVectors_[std::move(labelVectorPtr)];
}


void LabelVectorSet::visit(LabelVectorVisitor visitor) const {
    for (auto it = labelVectors_.cbegin(); it != labelVectors_.cend(); it++) {
        const auto& entry = *it;
        const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
        visitor(*labelVectorPtr);
    }
}
