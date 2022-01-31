#include "common/model/body_empty.hpp"


bool EmptyBody::covers(CContiguousConstView<const float32>::value_const_iterator begin,
                       CContiguousConstView<const float32>::value_const_iterator end) const {
    return true;
}

bool EmptyBody::covers(CsrConstView<const float32>::index_const_iterator indicesBegin,
                       CsrConstView<const float32>::index_const_iterator indicesEnd,
                       CsrConstView<const float32>::value_const_iterator valuesBegin,
                       CsrConstView<const float32>::value_const_iterator valuesEnd, float32* tmpArray1,
                       uint32* tmpArray2, uint32 n) const {
    return true;
}

void EmptyBody::visit(EmptyBodyVisitor emptyBodyVisitor, ConjunctiveBodyVisitor conjunctiveBodyVisitor) const {
    emptyBodyVisitor(*this);
}
