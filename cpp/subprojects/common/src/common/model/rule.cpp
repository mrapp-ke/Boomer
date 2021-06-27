#include "common/model/rule.hpp"


Rule::Rule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr)
    : bodyPtr_(std::move(bodyPtr)), headPtr_(std::move(headPtr)) {

}

const IBody& Rule::getBody() const {
    return *bodyPtr_;
}

const IHead& Rule::getHead() const {
    return *headPtr_;
}

void Rule::visit(IBody::EmptyBodyVisitor emptyBodyVisitor, IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                 IHead::FullHeadVisitor fullHeadVisitor, IHead::PartialHeadVisitor partialHeadVisitor) const {
    bodyPtr_->visit(emptyBodyVisitor, conjunctiveBodyVisitor);
    headPtr_->visit(fullHeadVisitor, partialHeadVisitor);
}
