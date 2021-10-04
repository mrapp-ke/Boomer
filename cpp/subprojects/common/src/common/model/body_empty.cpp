#include "common/model/body_empty.hpp"


bool EmptyBody::covers(CContiguousFeatureMatrix::const_iterator begin,
                       CContiguousFeatureMatrix::const_iterator end) const {
    return true;
}

bool EmptyBody::covers(CsrFeatureMatrix::index_const_iterator indicesBegin,
                       CsrFeatureMatrix::index_const_iterator indicesEnd,
                       CsrFeatureMatrix::value_const_iterator valuesBegin,
                       CsrFeatureMatrix::value_const_iterator valuesEnd, float32* tmpArray1, uint32* tmpArray2,
                       uint32 n) const {
    return true;
}

void EmptyBody::visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor) const {
    emptyBodyVisitor(*this);
}
